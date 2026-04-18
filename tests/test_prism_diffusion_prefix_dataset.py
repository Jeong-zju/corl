from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "main" / "scripts"))
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "main"
        / "policy"
        / "lerobot_policy_streaming_act"
        / "src"
    ),
)
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "main"
        / "policy"
        / "lerobot_policy_prism_diffusion"
        / "src"
    ),
)

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_policy_prism_diffusion.configuration_diffusion import PrismDiffusionConfig
from lerobot_policy_prism_diffusion.prefix_dataset import PrismDiffusionPrefixDataset
from lerobot_policy_streaming_act.prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
    prefix_image_key_from_camera_key,
)

CAMERA_KEY = "observation.images.main"
PREFIX_IMAGE_KEY = prefix_image_key_from_camera_key(CAMERA_KEY)


class _RowAccessor:
    def __getitem__(self, _idx):
        return type("Row", (), {"name": "toy-task"})()


class _TaskTable:
    iloc = _RowAccessor()


class _BaseDiffusionDataset:
    def __init__(self) -> None:
        self.hf_dataset = []
        for index in range(4):
            self.hf_dataset.append(
                {
                    "observation.state": torch.tensor(
                        [10.0 * (index + 1), 10.0 * (index + 1) + 1.0],
                        dtype=torch.float32,
                    ),
                    CAMERA_KEY: torch.full(
                        (3, 2, 2),
                        float(index + 1),
                        dtype=torch.float32,
                    ),
                    PATH_SIGNATURE_KEY: torch.tensor(
                        [100.0 + index, 200.0 + index],
                        dtype=torch.float32,
                    ),
                    DELTA_SIGNATURE_KEY: torch.tensor(
                        [900.0 + index, -900.0 - index],
                        dtype=torch.float32,
                    ),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(index),
                    "timestamp": torch.tensor(float(index)),
                    "task_index": torch.tensor(0),
                }
            )

        self.image_transforms = None
        self.features = {}
        self.delta_indices = {
            "observation.state": [-2, -1, 0],
            CAMERA_KEY: [-2, -1, 0],
            "action": [-2, -1, 0, 1],
        }
        self.meta = type(
            "Meta",
            (),
            {
                "camera_keys": (CAMERA_KEY,),
                "video_keys": (),
                "stats": {
                    "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    CAMERA_KEY: {"mean": [0.0], "std": [1.0]},
                    PATH_SIGNATURE_KEY: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    DELTA_SIGNATURE_KEY: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                },
                "episodes": [{"dataset_from_index": 0, "dataset_to_index": 4}],
                "tasks": _TaskTable(),
                "subtasks": None,
            },
        )()

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _ensure_hf_dataset_loaded(self) -> None:
        return None

    def _get_query_indices(self, abs_idx: int, ep_idx: int):
        episode = self.meta.episodes[ep_idx]
        ep_start = int(episode["dataset_from_index"])
        ep_end = int(episode["dataset_to_index"])
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in deltas]
            for key, deltas in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) or (abs_idx + delta >= ep_end) for delta in deltas]
            )
            for key, deltas in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_hf_dataset(self, query_indices):
        result = {}
        for key, indices in query_indices.items():
            if key == "action":
                result[key] = torch.stack(
                    [
                        torch.tensor(
                            [float(index), float(index) + 0.25],
                            dtype=torch.float32,
                        )
                        for index in indices
                    ],
                    dim=0,
                )
                continue

            result[key] = torch.stack(
                [self.hf_dataset[index][key] for index in indices],
                dim=0,
            )
        return result

    def __getitem__(self, idx: int):
        item = dict(self.hf_dataset[idx])
        ep_idx = int(item["episode_index"].item())
        abs_idx = int(item["index"].item())
        query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
        item.update(padding)
        item.update(self._query_hf_dataset(query_indices))
        item["task"] = "toy-task"
        return item


def test_prism_diffusion_prefix_dataset_aligns_current_step_without_mutating_diffusion_windows() -> None:
    dataset = PrismDiffusionPrefixDataset(
        _BaseDiffusionDataset(),
        prefix_train_max_steps=4,
        prefix_frame_stride=1,
        prefix_pad_value=-1.0,
        use_path_signature=True,
        use_delta_signature=True,
    )

    item = dataset[2]

    assert item["observation.state"].shape == (3, 2)
    assert torch.allclose(
        item["observation.state"],
        torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=torch.float32),
    )
    assert item[CAMERA_KEY].shape == (3, 3, 2, 2)
    assert torch.allclose(item[CAMERA_KEY][-1], torch.full((3, 2, 2), 3.0))
    assert item["action"].shape == (4, 2)

    assert item[PREFIX_STATE_KEY].shape == (4, 2)
    assert torch.allclose(
        item[PREFIX_STATE_KEY],
        torch.tensor(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [-1.0, -1.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert item[PREFIX_PATH_SIGNATURE_KEY].shape == (4, 2)
    assert torch.allclose(
        item[PREFIX_PATH_SIGNATURE_KEY][2],
        torch.tensor([102.0, 202.0], dtype=torch.float32),
    )
    assert item[PREFIX_DELTA_SIGNATURE_KEY].shape == (4, 2)
    assert torch.allclose(
        item[PREFIX_DELTA_SIGNATURE_KEY][0],
        torch.zeros(2, dtype=torch.float32),
    )
    assert torch.allclose(
        item[PREFIX_DELTA_SIGNATURE_KEY][2],
        torch.tensor([902.0, -902.0], dtype=torch.float32),
    )
    assert torch.equal(
        item[PREFIX_MASK_KEY],
        torch.tensor([True, True, True, False]),
    )
    assert item[PREFIX_IMAGE_KEY].shape == (4, 3, 2, 2)
    assert torch.allclose(item[PREFIX_IMAGE_KEY][2], torch.full((3, 2, 2), 3.0))

    # The last valid prefix element must align with diffusion's current decision
    # step t, i.e. the last observation in the native observation window.
    assert torch.allclose(item[PREFIX_STATE_KEY][2], item["observation.state"][-1])
    assert torch.allclose(item[PREFIX_IMAGE_KEY][2], item[CAMERA_KEY][-1])


def test_prism_diffusion_prefix_dataset_zeroes_first_selected_delta_signature() -> None:
    dataset = PrismDiffusionPrefixDataset(
        _BaseDiffusionDataset(),
        prefix_train_max_steps=1,
        prefix_frame_stride=1,
        prefix_pad_value=0.0,
        use_path_signature=False,
        use_delta_signature=True,
    )

    item = dataset[2]

    assert torch.equal(item[PREFIX_MASK_KEY], torch.tensor([True]))
    assert torch.allclose(
        item[PREFIX_STATE_KEY],
        torch.tensor([[30.0, 31.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        item[PREFIX_DELTA_SIGNATURE_KEY],
        torch.zeros((1, 2), dtype=torch.float32),
    )


def test_prism_diffusion_config_excludes_prefix_images_from_observation_cameras() -> None:
    cfg = PrismDiffusionConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            CAMERA_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            PREFIX_IMAGE_KEY: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(8, 3, 16, 16),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
    )

    assert tuple(cfg.image_features.keys()) == (CAMERA_KEY,)
