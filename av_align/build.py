"""build_layout：算子入口（方案 4.2）。

interventions 为空即原样通过——布局与无干预逐字段一致（性质测试 1 的 L1 部分；
与后端默认管线的逐字段等价由适配器在 P1-02 验证）。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from .interventions import Intervention, apply_intervention
from .layout import (
    AudioChunk,
    AudioTrack,
    FrameStamp,
    LayoutError,
    LayoutPlan,
    SCHEMA_ID,
    VideoTrack,
    compute_pairing,
    interleave,
)

__all__ = ["build_layout"]

_VERSION = "0.1.0"


def _validate_video(frames: Sequence[FrameStamp]) -> None:
    if not frames:
        return
    seen: set[int] = set()
    for f in frames:
        if f.index in seen:
            raise LayoutError(f"帧 index 重复: {f.index!r}")
        seen.add(f.index)
    # 视频帧允许时间槽重叠（变帧率/长曝光场景由调用方负责语义），只校验有限性与正时长
    for f in frames:
        if f.t_end <= f.t_start:
            raise LayoutError(f"帧 {f.index!r} 时长非正: [{f.t_start!r}, {f.t_end!r})")


def build_layout(
    video: VideoTrack,
    audio: AudioTrack,
    *,
    use_audio_in_video: bool = True,
    video_block_s: Optional[float] = None,
    interventions: Sequence[Intervention] = (),
    return_info: bool = False,
) -> Union[LayoutPlan, Tuple[LayoutPlan, dict[str, Any]]]:
    """构造显式音画时间布局。

    - 不修改输入轨道（帧/块对象全部按值重建）；
    - interventions 依次应用，顺序记录进布局；
    - use_audio_in_video=False 时音频与视频按 sequential 排布；
    - return_info=True 返回 (布局, 诊断)；诊断含逐侧统计与配对覆盖率，不含模型语义。
    """
    frames = [f for f in video.frames]
    chunks = [c for c in audio.chunks]
    applied: list[dict] = []
    for op in interventions:
        frames, chunks, rec = apply_intervention(op, frames, chunks)
        applied.append(rec)

    _validate_video(frames)

    mode = "interleaved" if use_audio_in_video else "sequential"
    seq = interleave(frames, chunks, mode=mode, video_block_s=video_block_s)
    pairing = compute_pairing(frames, chunks)

    plan = LayoutPlan(
        video={
            "fps": video.fps,
            "n_frames": len(frames),
            "frames": [
                {"index": f.index, "t_start": f.t_start, "t_end": f.t_end}
                for f in sorted(frames, key=lambda x: (x.t_start, x.index))
            ],
        },
        audio={
            "sample_rate": audio.sample_rate,
            "duration_s": audio.duration_or_infer(),
            "chunk_ms": audio.chunk_ms,
            "n_chunks": len(chunks),
            "chunks": [
                {
                    "index": c.index,
                    "t_start": c.t_start,
                    "t_end": c.t_end,
                    "sample_span": list(c.sample_span) if c.sample_span is not None else None,
                }
                for c in chunks
            ],
        },
        interleaving={"mode": mode, "video_block_s": video_block_s, "sequence": seq},
        pairing=pairing,
        interventions=applied,
        meta={"created_by": f"av_align {_VERSION}"},
    )
    plan.hash_ = plan.compute_hash()

    if not return_info:
        return plan
    n_frames = len(frames)
    n_chunks = len(chunks)
    paired_frames = sum(1 for p in pairing if p["audio_chunks"])
    info: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "n_frames": n_frames,
        "n_chunks": n_chunks,
        "n_sequence_tokens": len(seq),
        "n_pairing_entries": len(pairing),
        "frames_with_audio": paired_frames,
        "coverage_ratio": (paired_frames / n_frames) if n_frames else None,
        "unpaired_audio_chunks": n_chunks - len({c for p in pairing for c in p["audio_chunks"]}),
        "interventions_applied": applied,
        "hash": plan.hash_,
    }
    return plan, info
