"""干预算子（对齐研发方案 v2 §3.3）。

- `passthrough()`      原样通过，等价于空干预（哈希中性）；
- `shift_both(delta)`  两路目标区间加同一偏移（纳秒），配对保留；
- `reindex(side, mapping)`  按稳定单元 ID 改写指定模态的目标时间，保留原区间长度；
- `swap(side, pair)`   在两个同模态单元之间交换**内容归属**：两单元的目标时间区间互换
                        （内容随单元 ID 移动，源时间不变，逆变换可还原）。

多项干预按声明顺序应用并逐条记录（provenance）。目标区间是唯一被改写的量；
源时间恒保持输入值（v2：源时间与请求时间分离）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence, Union

from .layout import (
    TIME_EPS_NS,
    LayoutError,
    MediaUnit,
)


@dataclass(frozen=True)
class Passthrough:
    pass


@dataclass(frozen=True)
class ShiftBoth:
    delta_ns: int


@dataclass(frozen=True)
class ReindexSide:
    """mapping: unit_id → 新目标起点（纳秒 int）；未覆盖的单元 ID 报错（v2：无效映射明确报错）。"""

    side: str
    mapping: Mapping[str, int]


@dataclass(frozen=True)
class SwapUnits:
    """v2 `swap(side, pair)`：交换两单元的目标区间（内容归属随之互换）。"""

    side: str
    unit_id_a: str
    unit_id_b: str


Intervention = Union[Passthrough, ShiftBoth, ReindexSide, SwapUnits]

_SIDES = ("video", "audio")


def _side_units(side: str, video: list[MediaUnit], audio: list[MediaUnit]) -> list[MediaUnit]:
    if side == "video":
        return video
    if side == "audio":
        return audio
    raise LayoutError(f"未知模态: {side!r}（应为 'video' 或 'audio'）")


def apply_intervention(
    op: Intervention, video: list[MediaUnit], audio: list[MediaUnit]
) -> tuple[list[MediaUnit], list[MediaUnit], dict]:
    """应用单个干预到**目标区间**，返回新列表（不改输入）与 canonical 干预记录。"""
    if isinstance(op, Passthrough):
        return video, audio, {"op": "passthrough", "params": {}}
    if isinstance(op, ShiftBoth):
        if not isinstance(op.delta_ns, int) or isinstance(op.delta_ns, bool):
            raise LayoutError(f"shift_both 偏移必须是整数纳秒: {op.delta_ns!r}")
        return (
            [replace(u, t_start=u.t_start + op.delta_ns, t_end=u.t_end + op.delta_ns) for u in video],
            [replace(u, t_start=u.t_start + op.delta_ns, t_end=u.t_end + op.delta_ns) for u in audio],
            {"op": "shift_both", "params": {"delta_ns": op.delta_ns}},
        )
    if isinstance(op, ReindexSide):
        if op.side not in _SIDES:
            raise LayoutError(f"未知模态: {op.side!r}")
        units = _side_units(op.side, video, audio)
        ids = {u.unit_id for u in units}
        missing = set(op.mapping) - ids
        if missing:
            raise LayoutError(f"reindex mapping 引用不存在的单元 ID: {sorted(missing)!r}；必须全部有效")
        if callable(op.mapping):
            raise LayoutError("reindex mapping 必须是 unit_id → 纳秒 的显式映射（v2：无效映射明确报错）")
        remapped: dict[str, int] = {}
        for uid, new_start in op.mapping.items():
            if not isinstance(new_start, int) or isinstance(new_start, bool):
                raise LayoutError(f"reindex 目标起点必须是整数纳秒: {uid!r} → {new_start!r}")
            remapped[uid] = new_start
        new_units = []
        for u in units:
            if u.unit_id in remapped:
                dur = u.t_end - u.t_start
                new_units.append(replace(u, t_start=remapped[u.unit_id], t_end=remapped[u.unit_id] + dur))
            else:
                new_units.append(u)
        if op.side == "video":
            return new_units, audio, {"op": "reindex", "params": {"side": op.side, "unit_ids": sorted(remapped)}}
        return video, new_units, {"op": "reindex", "params": {"side": op.side, "unit_ids": sorted(remapped)}}
    if isinstance(op, SwapUnits):
        if op.side not in _SIDES:
            raise LayoutError(f"未知模态: {op.side!r}")
        units = _side_units(op.side, video, audio)
        by_id = {u.unit_id: u for u in units}
        for uid in (op.unit_id_a, op.unit_id_b):
            if uid not in by_id:
                raise LayoutError(f"swap 单元不存在: {uid!r}")
        a, b = by_id[op.unit_id_a], by_id[op.unit_id_b]
        swapped_a = replace(a, t_start=b.t_start, t_end=b.t_end)
        swapped_b = replace(b, t_start=a.t_start, t_end=a.t_end)
        new_units = [
            swapped_a if u.unit_id == op.unit_id_a else swapped_b if u.unit_id == op.unit_id_b else u
            for u in units
        ]
        record = {"op": "swap", "params": {"side": op.side, "pair": [op.unit_id_a, op.unit_id_b]}}
        if op.side == "video":
            return new_units, audio, record
        return video, new_units, record
    raise LayoutError(f"未知干预算子: {op!r}")
