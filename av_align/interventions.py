"""干预算子：语义先在 L1 定义，后端适配器只转发不携带自己的对齐逻辑（方案 4.3）。

四个算子，作用于时间槽而非内容：
- ShiftBoth(dt)         共同平移：两路同加 dt，等变对照用，预期不改变绑定；
- ReindexSide(side, m)  单路重索引：只改一路的时间起点，时长保持；主干预；
- SwapTimeSlots(side, a, b) 换位：交换两个时间槽的归属（内容随原 index 不动），
                          对应研究线的声音换位配对；
- Passthrough()         原样通过：identity 干预，检查算子接入本身无副作用。

每个算子返回 canonical 记录（进布局哈希），保证干预历史可复核。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional, Sequence, Union

from .layout import AudioChunk, FrameStamp, LayoutError, _check_finite

SIDE_VIDEO = "video"
SIDE_AUDIO = "audio"
_SIDES = (SIDE_VIDEO, SIDE_AUDIO)


@dataclass(frozen=True)
class ShiftBoth:
    dt: float


@dataclass(frozen=True)
class ReindexSide:
    """mapping: index -> 新 t_start（float 或 (old)->(new) 的 callable）。未覆盖的 index 报错。"""

    side: str
    mapping: Union[Mapping[int, float], Callable[[float], float]]


@dataclass(frozen=True)
class SwapTimeSlots:
    side: str
    index_a: int
    index_b: int


@dataclass(frozen=True)
class Passthrough:
    pass


Intervention = Union[ShiftBoth, ReindexSide, SwapTimeSlots, Passthrough]


def _shift_frames(frames: Sequence[FrameStamp], dt: float) -> list[FrameStamp]:
    return [replace(f, t_start=f.t_start + dt, t_end=f.t_end + dt) for f in frames]


def _shift_chunks(chunks: Sequence[AudioChunk], dt: float) -> list[AudioChunk]:
    return [replace(c, t_start=c.t_start + dt, t_end=c.t_end + dt) for c in chunks]


def _side_list(side: str, frames: list[FrameStamp], chunks: list[AudioChunk]):
    if side == SIDE_VIDEO:
        return frames
    if side == SIDE_AUDIO:
        return chunks
    raise LayoutError(f"未知模态: {side!r}（应为 'video' 或 'audio'）")


def _reindex(items, mapping):
    out: list = []
    for it in items:
        old_start = it.t_start
        if callable(mapping):
            new_start = float(mapping(old_start))
        else:
            if it.index not in mapping:
                raise LayoutError(f"重索引 mapping 未覆盖 index={it.index!r}；必须全覆盖，不静默保留")
            new_start = float(mapping[it.index])
        _check_finite("reindex new_start", new_start)
        dur = it.t_end - it.t_start
        out.append(replace(it, t_start=new_start, t_end=new_start + dur))
    return out


def apply_intervention(
    op: Intervention, frames: list[FrameStamp], chunks: list[AudioChunk]
) -> tuple[list[FrameStamp], list[AudioChunk], dict]:
    """应用单个干预，返回新列表（不改输入）与 canonical 干预记录。"""
    if isinstance(op, Passthrough):
        return frames, chunks, {"op": "passthrough", "params": {}}
    if isinstance(op, ShiftBoth):
        _check_finite("ShiftBoth.dt", op.dt)
        return (
            _shift_frames(frames, op.dt),
            _shift_chunks(chunks, op.dt),
            {"op": "shift_both", "params": {"dt": op.dt}},
        )
    if isinstance(op, ReindexSide):
        items = _side_list(op.side, frames, chunks)
        new_items = _reindex(items, op.mapping)
        if op.side == SIDE_VIDEO:
            return new_items, chunks, {"op": "reindex", "params": {"side": op.side}}
        return frames, new_items, {"op": "reindex", "params": {"side": op.side}}
    if isinstance(op, SwapTimeSlots):
        items = _side_list(op.side, frames, chunks)
        by_index = {it.index: it for it in items}
        for i in (op.index_a, op.index_b):
            if i not in by_index:
                raise LayoutError(f"换位 index 不存在: {i!r}")
        a, b = by_index[op.index_a], by_index[op.index_b]
        span_a, span_b = (a.t_start, a.t_end), (b.t_start, b.t_end)
        swapped_a = replace(a, t_start=span_b[0], t_end=span_b[1])
        swapped_b = replace(b, t_start=span_a[0], t_end=span_a[1])
        if op.side == SIDE_VIDEO:
            new_frames = [swapped_a if f.index == op.index_a else swapped_b if f.index == op.index_b else f for f in frames]
            return new_frames, chunks, {
                "op": "swap_time_slots",
                "params": {"side": op.side, "index_a": op.index_a, "index_b": op.index_b},
            }
        new_chunks = [swapped_a if c.index == op.index_a else swapped_b if c.index == op.index_b else c for c in chunks]
        return frames, new_chunks, {
            "op": "swap_time_slots",
            "params": {"side": op.side, "index_a": op.index_a, "index_b": op.index_b},
        }
    raise LayoutError(f"未知干预算子: {op!r}")
