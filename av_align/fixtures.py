"""合成事件夹具（方案 4.4 性质测试 5，亦供研究线复用）。

在已知时刻注入两类可判定事件：
- 画面：纯色闪帧（内容标记 = 该时间槽所属帧 index）；
- 声音：短促提示音（1 kHz 正弦短脉冲，标记 = 该时间槽所属音频块 index）。

L1 只消费时间槽；样本内容仅为下游（适配器/研究线）保留真实媒体语义。
无需 GPU、无需解码器。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import AudioTrack, VideoTrack


@dataclass
class SyntheticEventFixture:
    video: VideoTrack
    audio: AudioTrack
    audio_samples: np.ndarray
    event_t: float
    flash_frame_index: int
    beep_chunk_index: int
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

    # 注入短促提示音（内容标记；事件在媒体层可判定）
    beep_n = int(round(beep_ms / 1000.0 * sample_rate))
    beep_start = int(round((event_t - t0) * sample_rate))
    tt = np.arange(beep_n) / sample_rate
    end = min(beep_start + beep_n, n_samples)
    audio_samples[beep_start:end] += 0.5 * np.sin(2 * np.pi * beep_hz * tt[: end - beep_start])

    video = VideoTrack.from_fps(n_frames=int(round(duration_s * fps)), fps=fps, t0=t0)
    audio = AudioTrack.from_samples(audio_samples, sample_rate=sample_rate, chunk_ms=chunk_ms, t0=t0)

    flash_frame = next(f.index for f in video.frames if f.t_start <= event_t < f.t_end)
    beep_chunk = next(c.index for c in audio.chunks if c.t_start <= event_t < c.t_end)

    return SyntheticEventFixture(
        video=video,
        audio=audio,
        audio_samples=audio_samples,
        event_t=event_t,
        flash_frame_index=flash_frame,
        beep_chunk_index=beep_chunk,
        fps=fps,
        chunk_ms=chunk_ms,
        sample_rate=sample_rate,
        duration_s=duration_s,
    )


def add_beep_at(fix: SyntheticEventFixture, t: float, hz: float = 1200.0, beep_ms: float = 30.0) -> tuple[int, int]:
    """在时刻 t 追加第二个提示音，返回（覆盖 t 的帧 index, 覆盖 t 的块 index）。供多事件对照。"""
    if not (0 <= t < fix.duration_s):
        raise ValueError(f"t 必须在片段范围内: [0, {fix.duration_s})")
    beep_n = int(round(beep_ms / 1000.0 * fix.sample_rate))
    start = int(round(t * fix.sample_rate))
    tt = np.arange(beep_n) / fix.sample_rate
    end = min(start + beep_n, len(fix.audio_samples))
    fix.audio_samples[start:end] += 0.5 * np.sin(2 * np.pi * hz * tt[: end - start])
    flash_frame = next(f.index for f in fix.video.frames if f.t_start <= t < f.t_end)
    beep_chunk = next(c.index for c in fix.audio.chunks if c.t_start <= t < c.t_end)
    return flash_frame, beep_chunk
