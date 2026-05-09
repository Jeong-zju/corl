from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main"))
sys.path.insert(0, str(REPO_ROOT / "main" / "deploy"))

from deploy.config import (
    CommandConfig,
    DebugConfig,
    DeployConfig,
    ImageConfig,
    JointNameConfig,
    PolicyConfig,
    PolicyRTCConfig,
    RosConfig,
    RuntimeConfig,
    TopicConfig,
    load_deploy_config,
)
from deploy.gripper_hysteresis import GripperHysteresisConfig
import deploy.policy_runtime.loader as loader
from deploy.policy_runtime.loader import (
    PolicyRuntime,
    _import_lerobot_policy_submodule,
    _missing_dependency_error,
    apply_deploy_policy_overrides,
)


def _make_policy_config(**overrides) -> PolicyConfig:
    base = dict(
        type="streaming_act",
        path=Path("."),
        device="cpu",
        load_device=None,
        task="",
        n_action_steps=50,
        temporal_ensemble_coeff=0.0,
        state_dim=17,
        action_dim=17,
        arm_dof=7,
        base_action_dim=3,
        state_key="observation.state",
        action_key="action",
        image_keys={
            "left": "observation.images.left",
            "right": "observation.images.right",
            "top": "observation.images.top",
        },
        use_path_signature=False,
        use_delta_signature=False,
        signature_depth=1,
        signature_dim=None,
        signature_backend="simple",
        rtc=PolicyRTCConfig(),
    )
    base.update(overrides)
    return PolicyConfig(**base)


def _make_deploy_config(policy: PolicyConfig) -> DeployConfig:
    return DeployConfig(
        path=Path("deploy.yaml"),
        policy=policy,
        runtime=RuntimeConfig(control_hz=30.0),
        gripper_hysteresis=GripperHysteresisConfig(),
        debug=DebugConfig(
            enabled=False,
            publish_hz=0.0,
            attention_topic="/debug/attention",
            slot_memory_topic="/debug/slot_memory",
            signature_topic="/debug/signature",
            overlay_alpha=0.45,
            attention_query_step=0,
        ),
        image=ImageConfig(width=224, height=224, color_order="rgb"),
        ros=RosConfig(
            node_name="deploy_test",
            queue_size=1,
            topics=TopicConfig(
                image_left="/left",
                image_right="/right",
                image_top="/top",
                joint_state_left="/joint_left",
                joint_state_right="/joint_right",
                odom="/odom",
                cmd_vel="/cmd_vel",
                cmd_joint_left="/cmd_joint_left",
                cmd_joint_right="/cmd_joint_right",
            ),
            joint_names_left=JointNameConfig(name=[]),
            joint_names_right=JointNameConfig(name=[]),
        ),
        command=CommandConfig(
            publish_base=True,
            publish_arms=True,
            max_linear_x=1.0,
            max_linear_y=1.0,
            max_angular_z=1.0,
            deadzone_linear_x=0.0,
            deadzone_linear_y=0.0,
            deadzone_angular_z=0.0,
        ),
    )


def test_apply_deploy_policy_overrides_keeps_open_loop_when_coeff_is_zero() -> None:
    cfg = SimpleNamespace(
        n_action_steps=10,
        temporal_ensemble_coeff=None,
    )

    coeff, enabled = apply_deploy_policy_overrides(
        "act",
        cfg,
        _make_policy_config(),
    )

    assert coeff == 0.0
    assert enabled is False
    assert cfg.temporal_ensemble_coeff is None
    assert cfg.n_action_steps == 50


def test_apply_deploy_policy_overrides_forces_single_step_when_coeff_is_nonzero() -> None:
    cfg = SimpleNamespace(
        n_action_steps=25,
        temporal_ensemble_coeff=None,
    )

    coeff, enabled = apply_deploy_policy_overrides(
        "streaming_act",
        cfg,
        _make_policy_config(
            n_action_steps=50,
            temporal_ensemble_coeff=0.01,
        ),
    )

    assert coeff == 0.01
    assert enabled is True
    assert cfg.temporal_ensemble_coeff == 0.01
    assert cfg.n_action_steps == 1


def test_policy_runtime_first_frame_anchor_summary_reports_loaded_config() -> None:
    runtime = PolicyRuntime.__new__(PolicyRuntime)
    runtime.cfg = SimpleNamespace(
        use_first_frame_anchor=False,
        use_first_frame_anchor_in_slot_routing=True,
    )

    assert (
        runtime.first_frame_anchor_summary
        == "encoder_token=False, slot_routing=True"
    )


def test_policy_runtime_first_frame_anchor_summary_unknown_before_load() -> None:
    runtime = PolicyRuntime.__new__(PolicyRuntime)
    runtime.cfg = None

    assert runtime.first_frame_anchor_summary == "unknown"


def test_apply_deploy_policy_overrides_rejects_temporal_ensemble_for_rtc_policy() -> None:
    cfg = SimpleNamespace(
        n_action_steps=25,
        temporal_ensemble_coeff=None,
    )

    with pytest.raises(ValueError, match="not supported for policy type"):
        apply_deploy_policy_overrides(
            "pi05",
            cfg,
            _make_policy_config(
                type="pi05",
                n_action_steps=50,
                temporal_ensemble_coeff=0.01,
            ),
        )


@pytest.mark.parametrize("policy_type", ["pi0", "pi05", "smolvla"])
def test_policy_runtime_requires_vla_task_before_importing_lerobot(
    policy_type: str,
) -> None:
    runtime = PolicyRuntime(
        _make_deploy_config(
            _make_policy_config(
                type=policy_type,
                task="",
            )
        )
    )

    with pytest.raises(ValueError, match="policy.task"):
        runtime.load()


def test_load_deploy_config_rejects_temporal_ensemble_for_rtc_policy(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "deploy.yaml"
    config_path.write_text(
        """
policy:
  type: pi05
  path: /tmp/pi05
  device: cpu
  load_device: null
  task: return the book
  n_action_steps: 50
  temporal_ensemble_coeff: 0.01
  state_dim: 17
  action_dim: 17
  arm_dof: 7
  base_action_dim: 3
  state_key: observation.state
  action_key: action
  image_keys:
    left: observation.images.left
    right: observation.images.right
    top: observation.images.top
runtime:
  control_hz: 30.0
image:
  width: 224
  height: 224
  color_order: rgb
ros:
  node_name: deploy_test
  queue_size: 1
command:
  publish_base: true
  publish_arms: true
  max_linear_x: 1.0
  max_linear_y: 1.0
  max_angular_z: 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="temporal_ensemble_coeff"):
        load_deploy_config(config_path)


def test_load_deploy_config_accepts_pi05_without_temporal_ensemble(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "deploy.yaml"
    config_path.write_text(
        """
policy:
  type: pi05
  path: /tmp/pi05
  device: cpu
  load_device: null
  task: return the book
  n_action_steps: 50
  state_dim: 17
  action_dim: 17
  arm_dof: 7
  base_action_dim: 3
  state_key: observation.state
  action_key: action
  image_keys:
    left: observation.images.left
    right: observation.images.right
    top: observation.images.top
runtime:
  control_hz: 30.0
image:
  width: 224
  height: 224
  color_order: rgb
ros:
  node_name: deploy_test
  queue_size: 1
command:
  publish_base: true
  publish_arms: true
  max_linear_x: 1.0
  max_linear_y: 1.0
  max_angular_z: 1.0
""",
        encoding="utf-8",
    )

    cfg = load_deploy_config(config_path)
    assert cfg.policy.type == "pi05"
    assert cfg.policy.n_action_steps == 50
    assert cfg.policy.rtc.enabled is False


def test_load_deploy_config_accepts_pi05_rtc_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "deploy.yaml"
    config_path.write_text(
        """
policy:
  type: pi05
  path: /tmp/pi05
  device: cpu
  load_device: null
  task: return the book
  n_action_steps: 50
  state_dim: 17
  action_dim: 17
  arm_dof: 7
  base_action_dim: 3
  state_key: observation.state
  action_key: action
  image_keys:
    left: observation.images.left
    right: observation.images.right
    top: observation.images.top
  rtc:
    enabled: true
    prefix_attention_schedule: exp
    max_guidance_weight: 8.0
    execution_horizon: 12
    inference_delay_steps: 2
runtime:
  control_hz: 30.0
image:
  width: 224
  height: 224
  color_order: rgb
ros:
  node_name: deploy_test
  queue_size: 1
command:
  publish_base: true
  publish_arms: true
  max_linear_x: 1.0
  max_linear_y: 1.0
  max_angular_z: 1.0
""",
        encoding="utf-8",
    )

    cfg = load_deploy_config(config_path)
    assert cfg.policy.rtc.enabled is True
    assert cfg.policy.rtc.prefix_attention_schedule == "exp"
    assert cfg.policy.rtc.max_guidance_weight == 8.0
    assert cfg.policy.rtc.execution_horizon == 12
    assert cfg.policy.rtc.inference_delay_steps == 2


def test_missing_vla_dependency_error_names_nested_module() -> None:
    error = _missing_dependency_error(
        policy_name="SmolVLA",
        extra_name="smolvla",
        exc=ModuleNotFoundError("No module named 'transformers'", name="transformers"),
    )

    message = str(error)
    assert "`transformers`" in message
    assert "pip install -r requirements.txt" in message
    assert "lerobot[smolvla]==0.5.0" in message


def test_policy_runtime_load_quiets_transformers_loading_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_sequence: list[object] = []

    class FakeTransformersLogging:
        def __init__(self) -> None:
            self.verbosity = 11

        def get_verbosity(self) -> int:
            call_sequence.append(("get", self.verbosity))
            return self.verbosity

        def set_verbosity_error(self) -> None:
            call_sequence.append("set_error")
            self.verbosity = 40

        def set_verbosity(self, level: int) -> None:
            call_sequence.append(("set", level))
            self.verbosity = level

    fake_logging = FakeTransformersLogging()
    fake_transformers = ModuleType("transformers")
    fake_transformers.__path__ = []  # type: ignore[attr-defined]
    fake_transformers_utils = ModuleType("transformers.utils")
    fake_transformers_utils.logging = fake_logging
    fake_transformers.utils = fake_transformers_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_transformers_utils)

    dummy_cfg = SimpleNamespace(
        device=None,
        n_action_steps=8,
        temporal_ensemble_coeff=None,
        image_features={
            "observation.images.left": SimpleNamespace(shape=(3, 224, 224)),
        },
    )

    class DummyPreTrainedConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return dummy_cfg

    class DummyPolicy:
        def __init__(self, config):
            self.config = config
            self.to_calls: list[str] = []
            self.eval_called = False
            self.reset_called = False

        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert fake_logging.verbosity == 40
            return DummyPolicy(dummy_cfg)

        def to(self, device: str):
            self.to_calls.append(device)

        def eval(self):
            self.eval_called = True

        def reset(self):
            self.reset_called = True

    monkeypatch.setattr(
        loader,
        "_load_lerobot_pretrained_config_class",
        lambda policy_type: DummyPreTrainedConfig,
    )
    monkeypatch.setattr(
        loader,
        "load_pretrained_config_from_pretrained_dir",
        lambda config_cls, policy_dir, policy_label="policy": dummy_cfg,
    )
    monkeypatch.setattr(
        loader,
        "resolve_policy_dir",
        lambda policy_path: Path("/tmp/fake_policy"),
    )
    monkeypatch.setattr(
        loader,
        "resolve_deploy_policy_class",
        lambda policy_type, deploy_policy: DummyPolicy,
    )
    monkeypatch.setattr(
        loader,
        "_make_deploy_pre_post_processors",
        lambda **kwargs: ("pre", "post"),
    )

    runtime = PolicyRuntime(
        _make_deploy_config(
            _make_policy_config(
                type="smolvla",
                task="Return the book to its original location.",
                path=Path("/tmp/fake_policy"),
                device="cuda",
                load_device="cpu",
            )
        )
    )

    runtime.load()

    assert fake_logging.verbosity == 11
    assert ("set", 11) in call_sequence
    assert "set_error" in call_sequence
    assert runtime.policy is not None
    assert runtime.policy.to_calls == ["cuda"]
    assert runtime.policy.eval_called is True
    assert runtime.policy.reset_called is True


@pytest.mark.parametrize(
    ("policy_type", "module_file", "module_name"),
    [
        (
            "smolvla",
            "configuration_smolvla.py",
            "lerobot.policies.smolvla.configuration_smolvla",
        ),
    ],
)
def test_lerobot_policy_import_shim_skips_eager_policy_inits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_type: str,
    module_file: str,
    module_name: str,
) -> None:
    package_root = tmp_path / "lerobot"
    policy_root = package_root / "policies" / policy_type
    policy_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "policies" / "__init__.py").write_text(
        "raise RuntimeError('eager policies init ran')\n",
        encoding="utf-8",
    )
    (policy_root / "__init__.py").write_text(
        f"raise RuntimeError('eager {policy_type} init ran')\n",
        encoding="utf-8",
    )
    (policy_root / module_file).write_text(
        "SENTINEL = 'loaded directly'\n",
        encoding="utf-8",
    )

    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "lerobot" or name.startswith("lerobot.")
    }
    for name in original_modules:
        sys.modules.pop(name, None)

    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = _import_lerobot_policy_submodule(policy_type, module_name)
        assert module.SENTINEL == "loaded directly"
    finally:
        for name in list(sys.modules):
            if name == "lerobot" or name.startswith("lerobot."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
