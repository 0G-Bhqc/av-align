"""av_align.layout/1 — 显式音画时间布局 schema（冻结于 2026-11-15，G1）。

本模块是算子本体（L1 布局核心）的数据契约：
- 输入媒体流（帧时间戳 + 音频块时间戳），输出显式布局计划；
- 布局只描述时间组织，不解释模型语义（语义归后端适配器与研究线）；
- 所有时间量为 float64 秒；时间来源在 AudioChunk/FrameStamp 由调用方另行记录；
- 布局哈希对 canonical JSON 计算，供双后端对照与实验归档引用。

schema 字段改动必须升版本号（旧结果按旧 schema 读，禁止静默回填）。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

SCHEMA_ID = "av_align.layout/1"
HASH_ALGORITHM = "sha256-canonical-json-v1"
TIME_EPS = 1e-9
TIME_UNITS = "seconds_float64"


class LayoutError(ValueError):
    """输入或干预违反契约。按方案第 4.2 节，非法输入明确报错，不静默修正。"""


def _check_finite(name: str, *values: float) -> None:
    for v in values:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
            raise LayoutError(f"{name} 含非有限或非法时间值: {v!r}")


@dataclass(frozen=True)
class FrameStamp:
    """一帧视频的时间槽。index 是调用方的帧标识，布局内部不重排 index。"""

    index: int
    t_start: float
    t_end: float


@dataclass(frozen=True)
class AudioChunk:
    """一个音频 token 块的时间槽。sample_span 供回溯原始采样（闭开区间下标）。"""

    index: int
    t_start: float
    t_end: float
    sample_span: Optional[tuple[int, int]] = None


@dataclass
class VideoTrack:
    """帧序列与其时间点。帧可乱序输入（按 t_start 稳定排序，tie 以 index）。"""

    frames: list[FrameStamp]
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        if self.fps is not None:
            _check_finite("fps", self.fps)
            if self.fps <= 0:
                raise LayoutError(f"fps 必须为正: {self.fps!r}")

    @classmethod
    def from_timestamps(
        cls, times: Sequence[float], fps: Optional[float] = None, frame_duration: Optional[float] = None
    ) -> "VideoTrack":
        """由帧时间点构造。frame_duration 缺省时取 fps 的倒数；fps 也缺省则报错。"""
        if len(times) == 0:
            return cls(frames=[], fps=fps)
        if frame_duration is None:
            if fps is None:
                raise LayoutError("from_timestamps 需要 fps 或 frame_duration 之一以确定帧时长")
            frame_duration = 1.0 / float(fps)
        _check_finite("times", *times)
        _check_finite("frame_duration", frame_duration)
        if frame_duration <= 0:
            raise LayoutError(f"frame_duration 必须为正: {frame_duration!r}")
        frames = [FrameStamp(index=i, t_start=float(t), t_end=float(t) + frame_duration) for i, t in enumerate(times)]
        return cls(frames=frames, fps=fps)

    @classmethod
    def from_fps(cls, n_frames: int, fps: float, t0: float = 0.0) -> "VideoTrack":
        """均匀帧网格：第 i 帧覆盖 [t0 + i/fps, t0 + (i+1)/fps)。"""
        if n_frames < 0:
            raise LayoutError(f"n_frames 不能为负: {n_frames!r}")
        _check_finite("fps", fps, t0)
        if fps <= 0:
            raise LayoutError(f"fps 必须为正: {fps!r}")
        dur = 1.0 / float(fps)
        frames = [FrameStamp(index=i, t_start=t0 + i * dur, t_end=t0 + (i + 1) * dur) for i in range(n_frames)]
        return cls(frames=frames, fps=float(fps))

    def sorted_frames(self) -> list[FrameStamp]:
        return sorted(self.frames, key=lambda f: (f.t_start, f.index))


@dataclass
class AudioTrack:
    """音频块序列。块必须按时间升序输入（音频流是有序的）；乱序按契约报错。"""

    chunks: list[AudioChunk]
    sample_rate: Optional[float] = None
    duration_s: Optional[float] = None
    chunk_ms: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("sample_rate", "duration_s", "chunk_ms"):
            v = getattr(self, name)
            if v is not None:
                _check_finite(name, v)
                if v <= 0:
                    raise LayoutError(f"{name} 必须为正: {v!r}")
        prev_end: Optional[float] = None
        for c in self.chunks:
            _check_finite("chunk time", c.t_start, c.t_end)
            if c.t_end <= c.t_start:
                raise LayoutError(f"音频块 {c.index!r} 时长非正: [{c.t_start!r}, {c.t_end!r})")
            if prev_end is not None and c.t_start < prev_end - TIME_EPS:
                raise LayoutError("音频块必须按时间升序且不重叠输入；音频流是有序的，乱序请显式重建块")
            prev_end = c.t_end

    @classmethod
    def from_samples(cls, samples, sample_rate: float, chunk_ms: float, t0: float = 0.0) -> "AudioTrack":
        """由原始采样构造：按 chunk_ms 粒度切块，末块可短。samples 为 numpy 数组或序列，不被修改。"""
        _check_finite("sample_rate", sample_rate)
        _check_finite("chunk_ms", chunk_ms)
        if sample_rate <= 0 or chunk_ms <= 0:
            raise LayoutError("sample_rate 与 chunk_ms 必须为正")
        n = int(len(samples))
        dur = n / float(sample_rate)
        chunk_s = float(chunk_ms) / 1000.0
        chunks: list[AudioChunk] = []
        k = 0
        pos = 0
        while pos < n:
            span_end = min(pos + max(1, round(chunk_s * float(sample_rate))), n)
            chunks.append(
                AudioChunk(
                    index=k,
                    t_start=t0 + pos / float(sample_rate),
                    t_end=t0 + span_end / float(sample_rate),
                    sample_span=(pos, span_end),
                )
            )
            pos = span_end
            k += 1
        return cls(chunks=chunks, sample_rate=float(sample_rate), duration_s=dur, chunk_ms=float(chunk_ms))

    @classmethod
    def from_chunk_times(
        cls,
        times: Sequence[tuple[float, float]],
        sample_rate: Optional[float] = None,
        duration_s: Optional[float] = None,
        chunk_ms: Optional[float] = None,
    ) -> "AudioTrack":
        """由显式块时间构造（特征已按块给出，如后端音频特征帧）。times 为 (t_start, t_end) 对。"""
        chunks = [AudioChunk(index=i, t_start=float(s), t_end=float(e)) for i, (s, e) in enumerate(times)]
        return cls(chunks=chunks, sample_rate=sample_rate, duration_s=duration_s, chunk_ms=chunk_ms)

    def duration_or_infer(self) -> Optional[float]:
        if self.duration_s is not None:
            return self.duration_s
        if self.chunks:
            return self.chunks[-1].t_end
        return None


@dataclass
class LayoutPlan:
    """av_align.layout/1 计划对象。哈希覆盖除 hash 本身外的全部固定字段。"""

    video: dict[str, Any]
    audio: dict[str, Any]
    interleaving: dict[str, Any]
    pairing: list[dict[str, Any]]
    interventions: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)
    hash_: Optional[str] = None

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": SCHEMA_ID,
            "hash_algorithm": HASH_ALGORITHM,
            "time_units": TIME_UNITS,
            "video": self.video,
            "audio": self.audio,
            "interleaving": self.interleaving,
            "pairing": self.pairing,
            "interventions": self.interventions,
            "meta": self.meta,
        }
        if include_hash and self.hash_ is not None:
            d["hash"] = self.hash_
        return d

    def compute_hash(self) -> str:
        payload = json.dumps(self.to_dict(include_hash=False), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def canonical(self) -> dict[str, Any]:
        """返回带重算哈希的 canonical dict；与存储哈希不一致视为结果损坏。"""
        h = self.compute_hash()
        if self.hash_ is not None and h != self.hash_:
            raise LayoutError(f"布局哈希不一致: 存储 {self.hash_} != 重算 {h}")
        return self.to_dict(include_hash=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayoutPlan":
        if d.get("schema") != SCHEMA_ID:
            raise LayoutError(f"schema 不符: {d.get('schema')!r} != {SCHEMA_ID!r}（旧 schema 请按旧版本读取）")
        plan = cls(
            video=d["video"],
            audio=d["audio"],
            interleaving=d["interleaving"],
            pairing=d["pairing"],
            interventions=d["interventions"],
            meta=d.get("meta", {}),
            hash_=d.get("hash"),
        )
        plan.canonical()  # 读入即校验
        return plan


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """闭开区间重叠长度。触边（长度为 0）不算重叠。"""
    return min(a1, b1) - max(a0, b0)


def compute_pairing(frames: Sequence[FrameStamp], chunks: Sequence[AudioChunk]) -> list[dict[str, Any]]:
    """逐帧配对：与该帧时间槽重叠的音频块（重叠 > TIME_EPS）。

    这是绑定检验与干预定位的索引：配对表在共同平移下逐项不变（等变对照的依据）。
    """
    by_index = {c.index: c for c in chunks}
    out: list[dict[str, Any]] = []
    for f in sorted(frames, key=lambda x: (x.t_start, x.index)):
        paired: list[int] = []
        for c in sorted(chunks, key=lambda x: x.index):
            if overlap(f.t_start, f.t_end, c.t_start, c.t_end) > TIME_EPS:
                paired.append(c.index)
        out.append(
            {
                "video_frame": f.index,
                "audio_chunks": paired,
                "t_start": f.t_start,
                "t_end": f.t_end,
            }
        )
    return out


def interleave(
    frames: Sequence[FrameStamp],
    chunks: Sequence[AudioChunk],
    mode: str,
    video_block_s: Optional[float] = None,
    within_block_order: str = "audio_first",
) -> list[dict[str, Any]]:
    """生成媒体 token 序列。

    - mode="sequential"：全部视频帧（按 index）后接全部音频块（按 index）。
    - mode="interleaved"：
      - video_block_s=None：按时间逐事件合并；同刻平局——视频先于音频（schema 规则，
        平局以 tie_rank 保证确定性）。
      - video_block_s=Δ：按 Δ 秒值域分块（重述后端按秒块交错的打包方式）。
        within_block_order 决定块内顺序：
        - "audio_first"：时间严格序（音频早于块起点则先于视频块输出）；
        - "video_first"：每块先视频块后音频块（Transformers/MLX 后端的实际语义，
          P0-01/P0-02 勘察确认）。
    """
    if mode == "sequential":
        seq: list[dict[str, Any]] = []
        for f in sorted(frames, key=lambda x: x.index):
            seq.append({"modality": "video", "ref": f.index, "t_start": f.t_start, "t_end": f.t_end})
        for c in sorted(chunks, key=lambda x: x.index):
            seq.append({"modality": "audio", "ref": c.index, "t_start": c.t_start, "t_end": c.t_end})
        return seq
    if mode != "interleaved":
        raise LayoutError(f"未知交错模式: {mode!r}")

    if video_block_s is None:
        events = (
            [(f.t_start, 0, "video", f.index, f.t_start, f.t_end) for f in frames]
            + [(c.t_start, 1, "audio", c.index, c.t_start, c.t_end) for c in chunks]
        )
        events.sort(key=lambda e: (e[0], e[1], e[3]))
        return [
            {"modality": m, "ref": ref, "t_start": t0_, "t_end": t1_} for _, _, m, ref, t0_, t1_ in events
        ]

    # 分块交错：窗口 [t_first + jΔ, t_first + (j+1)Δ)；全体窗口 = 视频窗 ∪ 音频窗，升序处理
    _check_finite("video_block_s", video_block_s)
    if video_block_s <= 0:
        raise LayoutError(f"video_block_s 必须为正: {video_block_s!r}")
    if within_block_order not in ("audio_first", "video_first"):
        raise LayoutError(f"未知块内顺序: {within_block_order!r}")
    if not frames:
        return interleave([], chunks, "interleaved", None)
    t_first = min(f.t_start for f in frames)

    def _window_key(t: float) -> int:
        return int(math.floor((t - t_first) / video_block_s + TIME_EPS))

    video_windows: dict[int, list[FrameStamp]] = {}
    for f in frames:
        video_windows.setdefault(_window_key(f.t_start), []).append(f)
    audio_windows: dict[int, list[AudioChunk]] = {}
    for c in chunks:
        audio_windows.setdefault(_window_key(c.t_start), []).append(c)

    seq2: list[dict[str, Any]] = []
    for b in sorted(set(video_windows) | set(audio_windows)):
        wv = sorted(video_windows.get(b, []), key=lambda x: x.index)
        wa = sorted(audio_windows.get(b, []), key=lambda x: x.index)
        if within_block_order == "video_first":
            # 后端重述语义：每窗先视频块后音频块（Transformers/MLX 的值域分块行为）
            items: list[tuple[str, Any]] = [("video", f) for f in wv] + [("audio", c) for c in wa]
        else:
            # 同窗内时间严格序，平局视频先于音频（沿逐事件合并的 schema 规则）
            events = [(f.t_start, 0, "video", f) for f in wv] + [(c.t_start, 1, "audio", c) for c in wa]
            events.sort(key=lambda e: (e[0], e[1], e[3].index))
            items = [(m, it) for _, _, m, it in events]
        for m, item in items:
            seq2.append({"modality": m, "ref": item.index, "t_start": item.t_start, "t_end": item.t_end})
    return seq2
