"""av_align.layout/1 — 显式音画时间布局 schema（对齐研发方案 v2 §3.2，2026-11-15）。

契约要点（v2 落定）：
- 时间为共享媒体原点上的**整数纳秒**（int64），半开区间 [start, end)；
- 单元带**稳定单元 ID** 与**时间来源**标注（容器 PTS / 解码器生成 / 外推）；
- 布局区分**源时间**（media_units）与**请求/目标时间**（request.target，干预后）；
- swap 是**内容归属交换**：单元（内容随 ID）目标区间互换，源时间不变；
- 内容哈希覆盖 media_units + target + expected_sequence + pairing，**不含干预记录**
  （passthrough 与空干预哈希一致，v2 §4.1 性质 1）；干预记录为 provenance。
- 字段改动必须升版本号；旧结果按旧 schema 读，禁止静默回填。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

SCHEMA_ID = "av_align.layout/1"
HASH_ALGORITHM = "sha256-canonical-json-v1"
TIME_UNIT = "ns_int64"
NS = 1_000_000_000
TIME_EPS_NS = 1  # 1 纳秒：并列/触边判定下限
TIME_SOURCES = ("container_pts", "decoder", "extrapolated")
MODALITIES = ("video", "audio")


class LayoutError(ValueError):
    """输入或干预违反契约。非法输入明确报错，不静默修正（v2 §3.2）。"""


def s_to_ns(seconds: float) -> int:
    """秒（float）→ 整数纳秒。非有限值报错。"""
    if not math.isfinite(float(seconds)):
        raise LayoutError(f"非有限时间值: {seconds!r}")
    return int(round(float(seconds) * NS))


@dataclass(frozen=True)
class MediaUnit:
    """归一化媒体单元：稳定 ID + 模态 + 内容引用 + 源时间区间（纳秒）+ 时间来源。

    source_ref：视频为帧下标或 (首帧, 末帧)；音频为样本区间 (s0, e0)。内容引用由
    调用方维护，L1 不解释像素/波形。
    """

    unit_id: str
    modality: str
    t_start: int
    t_end: int
    time_source: str = "container_pts"
    source_ref: Optional[Any] = None

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise LayoutError("unit_id 不能为空")
        if self.modality not in MODALITIES:
            raise LayoutError(f"未知模态: {self.modality!r}")
        if self.time_source not in TIME_SOURCES:
            raise LayoutError(f"未知时间来源: {self.time_source!r}（须为 {TIME_SOURCES} 之一）")
        for name in ("t_start", "t_end"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise LayoutError(f"{name} 必须是整数纳秒: {v!r}")
        if self.t_end <= self.t_start:
            raise LayoutError(f"单元 {self.unit_id!r} 时长非正: [{self.t_start}, {self.t_end})")

    def with_span(self, t_start: int, t_end: int) -> "MediaUnit":
        return MediaUnit(self.unit_id, self.modality, t_start, t_end, self.time_source, self.source_ref)


@dataclass
class UnitTrack:
    """单模态归一化单元轨道。单元 ID 必须唯一；音频单元按时间升序且不重叠。"""

    units: list[MediaUnit]
    modality: str

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise LayoutError(f"未知模态: {self.modality!r}")
        ids: set[str] = set()
        prev_end: Optional[int] = None
        for u in self.units:
            if u.modality != self.modality:
                raise LayoutError(f"轨道模态不符: {self.modality!r} 内含 {u.modality!r}")
            if u.unit_id in ids:
                raise LayoutError(f"单元 ID 重复: {u.unit_id!r}")
            ids.add(u.unit_id)
            if self.modality == "audio" and prev_end is not None and u.t_start < prev_end - TIME_EPS_NS:
                raise LayoutError("音频单元必须按时间升序且不重叠；乱序请显式重建单元")
            prev_end = u.t_end

    def __len__(self) -> int:
        return len(self.units)

    # --- 工厂 ---

    @classmethod
    def from_frame_times(
        cls,
        times_s: Sequence[float],
        frame_duration_s: float,
        *,
        time_source: str = "container_pts",
        id_prefix: str = "v",
    ) -> "UnitTrack":
        """由帧时间点构造视频轨道。时间来源须显式（v2：缺失时间仅可按显式配置推断）。"""
        dur = s_to_ns(frame_duration_s)
        if dur <= 0:
            raise LayoutError(f"frame_duration_s 必须为正: {frame_duration_s!r}")
        units = [
            MediaUnit(
                unit_id=f"{id_prefix}{i:04d}",
                modality="video",
                t_start=s_to_ns(t),
                t_end=s_to_ns(t) + dur,
                time_source=time_source,
                source_ref=i,
            )
            for i, t in enumerate(times_s)
        ]
        return cls(units=units, modality="video")

    @classmethod
    def from_patch_grid(
        cls,
        n_patches: int,
        fps: float,
        temporal_patch_size: int = 2,
        *,
        t0_s: float = 0.0,
        time_source: str = "container_pts",
    ) -> "UnitTrack":
        """后端 patch 粒度视频轨道：第 p 个 patch 覆盖 [t0 + p·tps/fps, ...)（对齐 Transformers/MLX grid t 维）。"""
        if fps <= 0 or temporal_patch_size <= 0:
            raise LayoutError("fps 与 temporal_patch_size 必须为正")
        patch_dur_ns = s_to_ns(temporal_patch_size / float(fps))
        t0 = s_to_ns(t0_s)
        units = [
            MediaUnit(
                unit_id=f"p{p:04d}",
                modality="video",
                t_start=t0 + p * patch_dur_ns,
                t_end=t0 + (p + 1) * patch_dur_ns,
                time_source=time_source,
                source_ref=(p * temporal_patch_size, (p + 1) * temporal_patch_size),
            )
            for p in range(n_patches)
        ]
        return cls(units=units, modality="video")

    @classmethod
    def from_samples(
        cls,
        samples,
        sample_rate: float,
        chunk_ms: float,
        *,
        t0_s: float = 0.0,
        time_source: str = "decoder",
    ) -> "UnitTrack":
        """由原始采样构造音频轨道：按 chunk_ms 切块（末块可短），内容引用为样本区间。"""
        if sample_rate <= 0 or chunk_ms <= 0:
            raise LayoutError("sample_rate 与 chunk_ms 必须为正")
        n = int(len(samples))
        t0 = s_to_ns(t0_s)
        chunk_ns = s_to_ns(chunk_ms / 1000.0)
        chunk_samples = max(1, int(round(float(chunk_ms) / 1000.0 * float(sample_rate))))
        units: list[MediaUnit] = []
        k, pos = 0, 0
        while pos < n:
            span_end = min(pos + chunk_samples, n)
            units.append(
                MediaUnit(
                    unit_id=f"a{k:05d}",
                    modality="audio",
                    t_start=t0 + int(round(pos / float(sample_rate) * NS)),
                    t_end=t0 + int(round(span_end / float(sample_rate) * NS)),
                    time_source=time_source,
                    source_ref=(pos, span_end),
                )
            )
            pos = span_end
            k += 1
        return cls(units=units, modality="audio")

    @classmethod
    def from_unit_times(
        cls,
        spans_s: Sequence[tuple[float, float]],
        modality: str,
        *,
        time_source: str = "decoder",
        id_prefix: str = "u",
    ) -> "UnitTrack":
        """由显式 (t_start, t_end) 秒对构造（时间来源须显式）。"""
        units = [
            MediaUnit(
                unit_id=f"{id_prefix}{i:05d}",
                modality=modality,
                t_start=s_to_ns(s),
                t_end=s_to_ns(e),
                time_source=time_source,
            )
            for i, (s, e) in enumerate(spans_s)
        ]
        return cls(units=units, modality=modality)


def _canon_units(units: Sequence[MediaUnit]) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": u.unit_id,
            "modality": u.modality,
            "t_start": u.t_start,
            "t_end": u.t_end,
            "time_source": u.time_source,
            "source_ref": list(u.source_ref) if isinstance(u.source_ref, tuple) else u.source_ref,
        }
        for u in sorted(units, key=lambda x: (x.modality, x.t_start, x.unit_id))
    ]


def overlap_ns(a0: int, a1: int, b0: int, b1: int) -> int:
    """闭开区间重叠长度（纳秒）。触边（0）不算重叠（v2：端点相接视为无重叠）。"""
    return min(a1, b1) - max(a0, b0)


def compute_pairing(video_units: Sequence[MediaUnit], audio_units: Sequence[MediaUnit]) -> list[dict[str, Any]]:
    """逐视频单元配对：目标时间区间上重叠 > TIME_EPS_NS 的音频单元（稳定 ID 索引）。"""
    out: list[dict[str, Any]] = []
    for v in sorted(video_units, key=lambda x: (x.t_start, x.unit_id)):
        paired = [c.unit_id for c in sorted(audio_units, key=lambda x: x.unit_id)
                  if overlap_ns(v.t_start, v.t_end, c.t_start, c.t_end) > TIME_EPS_NS]
        out.append({"video_unit": v.unit_id, "audio_units": paired, "t_start": v.t_start, "t_end": v.t_end})
    return out


def interleave(
    video_units: Sequence[MediaUnit],
    audio_units: Sequence[MediaUnit],
    mode: str,
    video_block_s: Optional[float] = None,
    within_block_order: str = "audio_first",
) -> list[dict[str, Any]]:
    """生成 L1 期望媒体序列（对后端快照核验用）。

    - mode="sequential"：全部视频单元（按 ID）后接全部音频单元；
    - mode="interleaved"：
      - video_block_s=None：逐事件时间合并；平局视频先于音频（schema 规则）；
      - video_block_s=Δ：按 Δ 秒值域分窗（重述后端 2 秒窗打包）。
        within_block_order="video_first"（后端实际语义）或 "audio_first"（窗内时间严格）。
    """
    if mode == "sequential":
        seq: list[dict[str, Any]] = []
        for u in sorted(video_units, key=lambda x: x.unit_id):
            seq.append({"modality": "video", "unit_id": u.unit_id, "t_start": u.t_start, "t_end": u.t_end})
        for u in sorted(audio_units, key=lambda x: x.unit_id):
            seq.append({"modality": "audio", "unit_id": u.unit_id, "t_start": u.t_start, "t_end": u.t_end})
        return seq
    if mode != "interleaved":
        raise LayoutError(f"未知交错模式: {mode!r}")

    if video_block_s is None:
        events = (
            [(u.t_start, 0, "video", u) for u in video_units]
            + [(u.t_start, 1, "audio", u) for u in audio_units]
        )
        events.sort(key=lambda e: (e[0], e[1], e[3].unit_id))
        return [
            {"modality": m, "unit_id": u.unit_id, "t_start": u.t_start, "t_end": u.t_end}
            for _, _, m, u in events
        ]

    if video_block_s <= 0:
        raise LayoutError(f"video_block_s 必须为正: {video_block_s!r}")
    if within_block_order not in ("audio_first", "video_first"):
        raise LayoutError(f"未知块内顺序: {within_block_order!r}")
    if not video_units:
        return interleave([], audio_units, "interleaved", None)
    t_first = min(u.t_start for u in video_units)
    window_ns = int(round(video_block_s * NS))

    def _window_key(t: int) -> int:
        return int(math.floor((t - t_first) / window_ns + TIME_EPS_NS / window_ns))

    video_windows: dict[int, list[MediaUnit]] = {}
    for u in video_units:
        video_windows.setdefault(_window_key(u.t_start), []).append(u)
    audio_windows: dict[int, list[MediaUnit]] = {}
    for u in audio_units:
        audio_windows.setdefault(_window_key(u.t_start), []).append(u)

    seq2: list[dict[str, Any]] = []
    for b in sorted(set(video_windows) | set(audio_windows)):
        wv = sorted(video_windows.get(b, []), key=lambda x: x.unit_id)
        wa = sorted(audio_windows.get(b, []), key=lambda x: x.unit_id)
        if within_block_order == "video_first":
            items: list[tuple[str, MediaUnit]] = [("video", u) for u in wv] + [("audio", u) for u in wa]
        else:
            events = [(u.t_start, 0, "video", u) for u in wv] + [(u.t_start, 1, "audio", u) for u in wa]
            events.sort(key=lambda e: (e[0], e[1], e[3].unit_id))
            items = [(m, u) for _, _, m, u in events]
        for m, u in items:
            seq2.append({"modality": m, "unit_id": u.unit_id, "t_start": u.t_start, "t_end": u.t_end})
    return seq2


@dataclass
class LayoutPlan:
    """av_align.layout/1 计划对象。

    内容哈希覆盖 media_units（源）+ request.target + request.expected_sequence + pairing；
    request.interventions 为 provenance，**不在哈希内**（v2 §4.1：identity 与空干预同哈希）。
    """

    media_units: dict[str, list[dict[str, Any]]]
    request: dict[str, Any]
    pairing: list[dict[str, Any]]
    meta: dict[str, Any] = None  # type: ignore[assignment]
    hash_: Optional[str] = None

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "hash_algorithm": HASH_ALGORITHM,
            "time_units": TIME_UNIT,
            "media_units": self.media_units,
            "request": {
                "target": self.request["target"],
                "expected_sequence": self.request.get("expected_sequence", []),
            },
            "pairing": self.pairing,
        }

    def compute_hash(self) -> str:
        payload = json.dumps(self._hash_payload(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def canonical(self) -> dict[str, Any]:
        h = self.compute_hash()
        if self.hash_ is not None and h != self.hash_:
            raise LayoutError(f"布局哈希不一致: 存储 {self.hash_} != 重算 {h}")
        d = self._hash_payload()
        d["request"] = dict(self.request)
        d["meta"] = self.meta
        d["hash"] = self.hash_
        return d

    def to_dict(self) -> dict[str, Any]:
        return self.canonical()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayoutPlan":
        if d.get("schema") != SCHEMA_ID:
            raise LayoutError(f"schema 不符: {d.get('schema')!r} != {SCHEMA_ID!r}（旧 schema 请按旧版本读取）")
        plan = cls(
            media_units=d["media_units"],
            request=d["request"],
            pairing=d["pairing"],
            meta=d.get("meta", {}),
            hash_=d.get("hash"),
        )
        plan.canonical()
        return plan
