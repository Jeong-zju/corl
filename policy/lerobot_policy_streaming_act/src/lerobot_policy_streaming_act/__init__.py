"""Streaming ACT policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use lerobot_policy_streaming_act."
    ) from exc

from .configuration_streaming_act import FIRST_FRAME_ANCHOR_KEY, StreamingACTConfig
from .modeling_streaming_act import StreamingACTPolicy
from .prefix_sequence import (
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
)
from .processor_streaming_act import make_streaming_act_pre_post_processors

__all__ = [
    "FIRST_FRAME_ANCHOR_KEY",
    "PREFIX_MASK_KEY",
    "PREFIX_PATH_SIGNATURE_KEY",
    "PREFIX_STATE_KEY",
    "StreamingACTConfig",
    "StreamingACTPolicy",
    "make_streaming_act_pre_post_processors",
]
