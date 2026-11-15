"""build_layout：算子入口（对齐研发方案 v2 §3.1/§3.2）。

v2 入口：`build_layout(video_units, audio_units, *, interventions=(), return_info=False)`。
本实现额外接受布局选项（video_block_s / within_block_order），用于生成对后端打包的
L1 期望序列——这些选项属布局请求的一部分，不属模型配置（模型配置归 L2）。

- 源时间恒为输入单元的时间（不被修改）；干预只写目标区间；
- interventions 为空或全为 passthrough 时，目标区间与源时间逐字段一致（哈希一致，v2 §4.1）；
- return_info=True 返回 (布局, 诊断)：配对覆盖率、配对/配对失败计数、干预记录。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from .interventions import Intervention, apply_intervention
from .layout import (
    SCHEMA_ID,
    LayoutError,
    LayoutPlan,
    MediaUnit,
    UnitTrack,
    _canon_units,
    compute_pairing,
    interleave,
)

__all__ = ["build_layout"]

_VERSION = "0.2.0"


def build_layout(
    video_units: UnitTrack,
    audio_units: UnitTrack,
    *,
    interventions: Sequence[Intervention] = (),
    use_audio_in_video: bool = True,
    video_block_s: Optional[float] = None,
    within_block_order: str = "audio_first",
    expected_sequence: bool = True,
    return_info: bool = False,
) -> Union[dict, Tuple[dict, dict[str, Any]]]:
    """构造显式音画时间布局（av_align.layout/1）。

    - 不修改输入轨道（单元对象按值重建）；
    - 干预依次应用到目标区间，顺序记录进 request.interventions（provenance）；
    - video_block_s / within_block_order 控制 L1 期望序列的分块方式（后端重述用）；
    - expected_sequence=False 可省略序列（只做配对布局）。
    """
    if video_units.modality != "video" or audio_units.modality != "audio":
        raise LayoutError("build_layout 需要一个视频轨道与一个音频轨道")

    src_video: list[MediaUnit] = list(video_units.units)
    src_audio: list[MediaUnit] = list(audio_units.units)
    tgt_video, tgt_audio = list(src_video), list(src_audio)
    applied: list[dict] = []
    for op in interventions:
        tgt_video, tgt_audio, rec = apply_intervention(op, tgt_video, tgt_audio)
        applied.append(rec)

    mode = "interleaved" if use_audio_in_video else "sequential"
    seq = (
        interleave(
            tgt_video, tgt_audio, mode=mode, video_block_s=video_block_s, within_block_order=within_block_order
        )
        if expected_sequence
        else []
    )
    pairing = compute_pairing(tgt_video, tgt_audio)

    plan_media_units = {"video": _canon_units(src_video), "audio": _canon_units(src_audio)}
    request = {
        "interventions": applied,
        "target": {"video": _canon_units(tgt_video), "audio": _canon_units(tgt_audio)},
        "expected_sequence": seq,
        "sequence_opts": {
            "video_block_s": video_block_s,
            "within_block_order": within_block_order if video_block_s is not None else None,
        },
    }
    plan = LayoutPlan(media_units=plan_media_units, request=request, pairing=pairing)
    plan.meta = {"created_by": f"av_align {_VERSION}"}
    plan.hash_ = plan.compute_hash()

    if not return_info:
        return plan.canonical()
    n_v, n_a = len(tgt_video), len(tgt_audio)
    paired_v = sum(1 for p in pairing if p["audio_units"])
    info: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "n_video_units": n_v,
        "n_audio_units": n_a,
        "n_sequence_tokens": len(seq),
        "n_pairing_entries": len(pairing),
        "video_units_with_audio": paired_v,
        "coverage_ratio": (paired_v / n_v) if n_v else None,
        "unpaired_audio_units": n_a - len({uid for p in pairing for uid in p["audio_units"]}),
        "interventions_applied": applied,
        "hash": plan.hash_,
    }
    return plan.canonical(), info
