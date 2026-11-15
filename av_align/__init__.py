"""av_align：音画时间对齐算子（L1 布局核心 + 干预算子 + 后端适配器）。

对齐研发方案 v2（§3 契约）：整数纳秒时间、稳定单元 ID + 时间来源、源时间/目标时间分离、
内容归属 swap、identity 干预哈希中性。公开入口：
- build_layout(video_units, audio_units, ...) -> av_align.layout/1 canonical dict
- 干预算子：Passthrough / ShiftBoth / ReindexSide / SwapUnits
- 单元轨道：UnitTrack（from_frame_times / from_patch_grid / from_samples / from_unit_times）
- 合成事件夹具：make_synthetic_event_fixture（无需 GPU/解码器）
"""

from .build import build_layout
from .fixtures import SyntheticEventFixture, add_beep_at, make_synthetic_event_fixture
from .interventions import (
    Intervention,
    Passthrough,
    ReindexSide,
    ShiftBoth,
    SwapUnits,
    apply_intervention,
)
from .layout import (
    HASH_ALGORITHM,
    NS,
    SCHEMA_ID,
    TIME_UNIT,
    LayoutError,
    LayoutPlan,
    MediaUnit,
    UnitTrack,
    compute_pairing,
    interleave,
    s_to_ns,
)

__version__ = "0.2.0"

__all__ = [
    "build_layout",
    "SyntheticEventFixture",
    "add_beep_at",
    "make_synthetic_event_fixture",
    "Intervention",
    "Passthrough",
    "ReindexSide",
    "ShiftBoth",
    "SwapUnits",
    "apply_intervention",
    "SCHEMA_ID",
    "HASH_ALGORITHM",
    "TIME_UNIT",
    "NS",
    "s_to_ns",
    "LayoutError",
    "LayoutPlan",
    "MediaUnit",
    "UnitTrack",
    "compute_pairing",
    "interleave",
    "__version__",
]
