"""Streaming ACT policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use lerobot_policy_streaming_act."
    ) from exc

from .configuration_act import StreamingACTConfig
from .modeling_act import StreamingACTPolicy
from .processor_act import make_act_pre_post_processors

make_streaming_act_pre_post_processors = make_act_pre_post_processors

__all__ = [
    "StreamingACTConfig",
    "StreamingACTPolicy",
    "make_act_pre_post_processors",
    "make_streaming_act_pre_post_processors",
]
