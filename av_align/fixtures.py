"""合成事件夹具（v2 §4.1 事件配对性质 + 研究线复用）。

已知时刻注入两类可判定事件：
- 画面：纯色闪帧（帧下标标记；时间来源 container_pts，均匀网格）；
- 声音：短促提示音（1 kHz 正弦短脉冲，内容引用为样本区间）。

L1 只消费时间区间；样本内容仅为下游（适配器/研究线）保留真实媒体语义。
无需 GPU、无需解码器。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import NS, UnitTrack, s_to_ns


@dataclass
class SyntheticEventFixture:
    video: UnitTrack
    audio: UnitTrack
    audio_samples: np.ndarray
    event_t_s: float
    flash_unit_id: str
    beep_unit_id: str
    fps: float
    chunk_ms: float
    sample_rate: float
    duration_s: float


def make_synthetic_event_fixture(
    duration_s: float = 10.0,
    fps: float = 4.0,
    sample_rate: float = 16000.0,
    chunk_ms: float = 40.0,
    event_t: float = 3.2,
    t0: float = 0.0,
    beep_hz: float = 1000.0,
    beep_ms: float = 30.0,
    seed: int = 0,
) -> SyntheticEventFixture:
    """构造单事件合成片段：闪帧与提示音共享同一事件时刻，天然落在相同时间槽内。"""
    if not (t0 <= event_t < t0 + duration_s):
        raise ValueError(f"event_t 必须在片段范围内: [{t0}, {t0 + duration_s})")
    if fps <= 0 or sample_rate <= 0 or chunk_ms <= 0:
        raise ValueError("fps / sample_rate / chunk_ms 必须为正")

    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_s * sample_rate))
    audio_samples = (rng.standard_normal(n_samples) * 0.01).astype(np.float64)

    beep_n = int(round(beep_ms / 1000.0 * sample_rate))
    beep_start = int(round((event_t - t0) * sample_rate))
    tt = np.arange(beep_n) / sample_rate
    end = min(beep_start + beep_n, n_samples)
    audio_samples[beep_start:end] += 0.5 * np.sin(2 * np.pi * beep_hz * tt[: end - beep_start])

    video = UnitTrack.from_frame_times(
        [t0 + i / fps for i in range(int(round(duration_s * fps)))],
        frame_duration_s=1.0 / fps,
        time_source="container_pts",
    )
    audio = UnitTrack.from_samples(audio_samples, sample_rate=sample_rate, chunk_ms=chunk_ms, t0_s=t0, time_source="decoder")

    ev_ns = s_to_ns(event_t)
    flash_unit = next(u.unit_id for u in video.units if u.t_start <= ev_ns < u.t_end)
    beep_unit = next(u.unit_id for u in audio.units if u.t_start <= ev_ns < u.t_end)

    return SyntheticEventFixture(
        video=video,
        audio=audio,
        audio_samples=audio_samples,
        event_t_s=event_t,
        flash_unit_id=flash_unit,
        beep_unit_id=beep_unit,
        fps=fps,
        chunk_ms=chunk_ms,
        sample_rate=sample_rate,
        duration_s=duration_s,
    )


def add_beep_at(fix: SyntheticEventFixture, t: float, hz: float = 1200.0, beep_ms: float = 30.0) -> tuple[str, str]:
    """在时刻 t 追加第二个提示音，返回（覆盖 t 的视频单元 ID, 覆盖 t 的音频单元 ID）。"""
    if not (0 <= t < fix.duration_s):
        raise ValueError(f"t 必须在片段范围内: [0, {fix.duration_s})")
    beep_n = int(round(beep_ms / 1000.0 * fix.sample_rate))
    start = int(round(t * fix.sample_rate))
    tt = np.arange(beep_n) / fix.sample_rate
    end = min(start + beep_n, len(fix.audio_samples))
    fix.audio_samples[start:end] += 0.5 * np.sin(2 * np.pi * hz * tt[: end - start])
    # 追加样本不改变既有单元区间；新内容超出原轨道末尾时调用方须重建音频轨道
    ev_ns = s_to_ns(t)
    flash_unit = next(u.unit_id for u in fix.video.units if u.t_start <= ev_ns < u.t_end)
    beep_unit = next((u.unit_id for u in fix.audio.units if u.t_start <= ev_ns < u.t_end), None)
    if beep_unit is None:
        raise ValueError(f"时刻 {t}s 超出既有音频轨道范围；请重建轨道后再调用")
    return flash_unit, beep_unit
