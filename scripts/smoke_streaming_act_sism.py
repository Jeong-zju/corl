from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def ensure_imports(repo_root: Path) -> None:
    scripts_root = repo_root / "main" / "scripts"
    streaming_act_src = repo_root / "main" / "policy" / "lerobot_policy_streaming_act" / "src"
    sys.path.insert(0, str(scripts_root))
    sys.path.insert(0, str(streaming_act_src))


def feat(ftype, shape):
    from lerobot.configs.types import PolicyFeature

    return PolicyFeature(type=ftype, shape=shape)


def build_streaming_act_config(
    *,
    state_dim: int,
    signature_dim: int,
    image_size: int,
    prefix_steps: int,
    action_dim: int,
    memory_mode: str,
):
    from lerobot.configs.types import FeatureType
    from lerobot.utils.constants import ACTION
    from lerobot_policy_streaming_act.configuration_streaming_act import StreamingACTConfig

    input_features = {
        "observation.state": feat(FeatureType.STATE, (state_dim,)),
        "observation.images.front": feat(FeatureType.VISUAL, (3, image_size, image_size)),
        "observation.path_signature": feat(FeatureType.STATE, (signature_dim,)),
        "observation.delta_signature": feat(FeatureType.STATE, (signature_dim,)),
        "observation.prefix_state": feat(FeatureType.STATE, (prefix_steps, state_dim)),
        "observation.prefix_path_signature": feat(
            FeatureType.STATE, (prefix_steps, signature_dim)
        ),
        "observation.prefix_delta_signature": feat(
            FeatureType.STATE, (prefix_steps, signature_dim)
        ),
        "observation.prefix_mask": feat(FeatureType.STATE, (prefix_steps,)),
        "observation.prefix_images.front": feat(
            FeatureType.VISUAL, (prefix_steps, 3, image_size, image_size)
        ),
    }
    output_features = {ACTION: feat(FeatureType.ACTION, (action_dim,))}

    common = dict(
        device="cpu",
        use_amp=False,
        push_to_hub=False,
        pretrained_backbone_weights=None,
        input_features=input_features,
        output_features=output_features,
        chunk_size=4,
        use_vae=False,
        use_path_signature=True,
        use_delta_signature=True,
        signature_dim=signature_dim,
        signature_depth=3,
        signature_hidden_dim=32,
        use_prefix_sequence_training=True,
        prefix_train_max_steps=prefix_steps,
        prefix_frame_stride=1,
        prefix_pad_value=0.0,
        use_visual_prefix_memory=True,
        use_memory_conditioned_encoder_film=True,
    )
    if memory_mode == "gru":
        return StreamingACTConfig(
            **common,
            n_action_steps=2,
            num_memory_slots=2,
            use_signature_conditioned_visual_prefix_memory=True,
        )
    if memory_mode == "sism":
        return StreamingACTConfig(
            **common,
            n_action_steps=1,
            use_signature_indexed_slot_memory=True,
            slot_memory_num_slots=3,
            slot_memory_routing_hidden_dim=24,
            slot_memory_use_delta_routing=True,
            slot_memory_use_softmax_routing=True,
            slot_memory_use_readout_pooling=True,
            slot_memory_balance_loss_coef=0.1,
            slot_memory_consistency_loss_coef=0.2,
        )
    raise ValueError(f"Unsupported memory_mode={memory_mode!r}.")


def run_synthetic(memory_modes: list[str]) -> None:
    from lerobot.utils.constants import ACTION
    from lerobot_policy_streaming_act.modeling_streaming_act import StreamingACT, StreamingACTPolicy

    torch.manual_seed(0)
    prefix_steps = 4
    state_dim = 5
    signature_dim = 8
    action_dim = 3
    image_size = 64

    def make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
        batch = {
            "observation.state": torch.randn(batch_size, state_dim),
            "observation.images.front": torch.randn(batch_size, 3, image_size, image_size),
            "observation.path_signature": torch.randn(batch_size, signature_dim),
            "observation.delta_signature": torch.randn(batch_size, signature_dim),
            "observation.prefix_state": torch.randn(batch_size, prefix_steps, state_dim),
            "observation.prefix_path_signature": torch.randn(
                batch_size, prefix_steps, signature_dim
            ),
            "observation.prefix_delta_signature": torch.randn(
                batch_size, prefix_steps, signature_dim
            ),
            "observation.prefix_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 0]],
                dtype=torch.bool,
            ),
            "observation.prefix_images.front": torch.randn(
                batch_size, prefix_steps, 3, image_size, image_size
            ),
            ACTION: torch.randn(batch_size, 4, action_dim),
            "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        }
        batch["observation.prefix_state"][1, 3] = 0
        batch["observation.prefix_path_signature"][1, 3] = 0
        batch["observation.prefix_delta_signature"][1, 3] = 0
        batch["observation.prefix_images.front"][1, 3] = 0
        batch["observation.images"] = [batch["observation.images.front"]]
        return batch

    for memory_mode in memory_modes:
        cfg = build_streaming_act_config(
            state_dim=state_dim,
            signature_dim=signature_dim,
            image_size=image_size,
            prefix_steps=prefix_steps,
            action_dim=action_dim,
            memory_mode=memory_mode,
        )
        model = StreamingACT(cfg)
        batch = make_batch()
        actions, _ = model(batch)
        online_state, _ = model.compute_online_visual_prefix_memory_token(
            {
                "observation.images": [batch["observation.images.front"]],
                "observation.state": batch["observation.state"],
                "observation.path_signature": batch["observation.path_signature"],
                "observation.delta_signature": batch["observation.delta_signature"],
            }
        )
        print(
            {
                "mode": memory_mode,
                "actions_shape": tuple(actions.shape),
                "online_memory_shape": tuple(online_state.shape),
                "aux_losses": {
                    key: round(float(value.detach().cpu()), 6)
                    for key, value in model.get_visual_prefix_memory_aux_losses().items()
                },
            }
        )

        if memory_mode == "sism":
            policy = StreamingACTPolicy(cfg)
            policy.train()
            loss, loss_dict = policy.forward(make_batch())
            print(
                {
                    "mode": memory_mode,
                    "loss": round(float(loss.detach().cpu()), 6),
                    "slot_memory_loss_keys": sorted(
                        key for key in loss_dict if key.startswith("slot_memory")
                    ),
                }
            )


def run_braidedhub_rollout(steps: int) -> None:
    from env.braidedhub_env import (
        BraidedHub2DEnv,
        build_default_map_config,
        make_lerobot_base_image,
        render_lerobot_frame,
    )
    from eval_helpers import compute_delta_signature_step_np, compute_simple_signature_np
    from lerobot_policy_streaming_act.modeling_streaming_act import StreamingACTPolicy

    torch.manual_seed(0)
    np.random.seed(0)
    sig_depth = 3
    state_dim = 2
    signature_dim = state_dim * sig_depth
    image_size = 128
    cfg = build_streaming_act_config(
        state_dim=state_dim,
        signature_dim=signature_dim,
        image_size=image_size,
        prefix_steps=4,
        action_dim=2,
        memory_mode="sism",
    )
    policy = StreamingACTPolicy(cfg)
    policy.reset()

    map_config = build_default_map_config()
    base_image = make_lerobot_base_image(map_config, image_size=image_size)
    env = BraidedHub2DEnv(map_config=map_config, enable_randomize=False)
    state = env.reset(task_id=0)
    state_history: list[np.ndarray] = []
    previous_signature = None

    for step in range(steps):
        state_vec = np.asarray(state, dtype=np.float32)
        state_history.append(state_vec.copy())
        signature = compute_simple_signature_np(
            np.stack(state_history, axis=0),
            sig_depth=sig_depth,
        )
        delta_signature = compute_delta_signature_step_np(signature, previous_signature)
        previous_signature = signature.copy()
        frame = render_lerobot_frame(base_image, map_config, robot_xy=tuple(state_vec.tolist()))
        batch = {
            "observation.state": torch.from_numpy(state_vec).unsqueeze(0),
            "observation.images.front": torch.from_numpy(frame)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .unsqueeze(0)
            / 255.0,
            "observation.path_signature": torch.from_numpy(signature).unsqueeze(0),
            "observation.delta_signature": torch.from_numpy(delta_signature).unsqueeze(0),
        }
        action = policy.select_action(batch)[0].detach().cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        state, reward, done, _ = env.step((float(action[0]), float(action[1])))
        print(
            {
                "step": step,
                "action": [round(float(x), 4) for x in action.tolist()],
                "state": [round(float(x), 4) for x in state],
                "reward": round(float(reward), 4),
                "done": bool(done),
                "memory_debug": policy.get_visual_prefix_memory_debug_stats(),
            }
        )
        if done:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Streaming ACT SISM smoke tests.")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "braidedhub_rollout"],
        required=True,
    )
    parser.add_argument(
        "--memory-mode",
        choices=["gru", "sism", "both"],
        default="both",
    )
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ensure_imports(repo_root)

    if args.mode == "synthetic":
        memory_modes = ["gru", "sism"] if args.memory_mode == "both" else [args.memory_mode]
        run_synthetic(memory_modes)
        return

    run_braidedhub_rollout(steps=max(1, int(args.steps)))


if __name__ == "__main__":
    main()
