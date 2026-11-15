"""av_align：音画时间对齐算子（L1 布局核心 + 干预算子 + 后端适配器）。

公开入口：
- build_layout(video, audio, ...) -> LayoutPlan（av_align.layout/1）
- 干预算子：ShiftBoth / ReindexSide / SwapTimeSlots / Passthrough
- 合成事件夹具：make_synthetic_event_fixture（无需 GPU/解码器）
"""

from .build import build_layout
from .fixtures import SyntheticEventFixture, add_beep_at, make_synthetic_event_fixture
from .interventions import (
    Intervention,
    Passthrough,
    ReindexSide,
    ShiftBoth,
    SwapTimeSlots,
    apply_intervention,
)
from .layout import (
    SCHEMA_ID,
    AudioChunk,
    AudioTrack,
    FrameStamp,
    LayoutError,
    LayoutPlan,
    VideoTrack,
    compute_pairing,
    interleave,
)

__version__ = "0.1.0"

__all__ = [
    "build_layout",
    "SyntheticEventFixture",
    "add_beep_at",
    "make_synthetic_event_fixture",
    "Intervention",
    "Passthrough",
    "ReindexSide",
    "ShiftBoth",
    "SwapTimeSlots",
    "apply_intervention",
    "SCHEMA_ID",
    "AudioChunk",
    "AudioTrack",
    "FrameStamp",
    "LayoutError",
    "LayoutPlan",
    "VideoTrack",
    "compute_pairing",
    "interleave",
    "__version__",
]
