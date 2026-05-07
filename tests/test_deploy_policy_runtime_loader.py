from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    RosConfig,
    RuntimeConfig,
    TopicConfig,
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
        groot_attn_implementation="eager",
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

    coeff, enabled = apply_deploy_policy_overrides(cfg, _make_policy_config())

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


@pytest.mark.parametrize("policy_type", ["pi05", "smolvla"])
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


def test_missing_pi05_dependency_error_uses_pi_extra() -> None:
    error = _missing_dependency_error(
        policy_name="PI05",
        extra_name="pi",
        exc=ModuleNotFoundError("No module named 'transformers'", name="transformers"),
    )

    message = str(error)
    assert "`transformers`" in message
    assert "pip install -r requirements.txt" in message
    assert "lerobot[pi]==0.5.0" in message


def test_groot_deploy_compatibility_patches_include_state_dict_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    fake_train_policy = ModuleType("train_policy")

    def _record(name: str):
        def _fn(*args, **kwargs):
            calls.append(name)

        return _fn

    fake_train_policy.install_groot_attention_implementation_patch = _record(
        "attention"
    )
    fake_train_policy.install_groot_meta_tensor_compatibility_patch = _record("meta")
    fake_train_policy.install_groot_transformers_loading_compatibility_patch = _record(
        "transformers"
    )
    fake_train_policy.install_groot_processor_tensor_compatibility_patch = _record(
        "processor"
    )
    fake_train_policy.install_groot_state_dict_compatibility_patch = _record(
        "state_dict"
    )

    monkeypatch.setitem(sys.modules, "train_policy", fake_train_policy)

    loader._install_groot_deploy_compatibility_patches("eager")

    assert calls == [
        "attention",
        "meta",
        "transformers",
        "processor",
        "state_dict",
    ]


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
                type="groot",
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
            "pi05",
            "configuration_pi05.py",
            "lerobot.policies.pi05.configuration_pi05",
        ),
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
