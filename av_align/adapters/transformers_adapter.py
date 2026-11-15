"""后端 A 适配器：Transformers / Qwen2.5-Omni（只读重述，方案 5.2）。

原则：位置与交错全部调用后端自身代码（processor 的 replace_video_with_audio_token、
模型基类的 get_rope_index）；本模块只做三件事：
1. 把 L1 布局计划换算成后端参数（video_block_s=seconds_per_chunk、块内视频先行）；
2. 构造后端可直接消费的合成输入表（无需媒体容器与模型权重）；
3. 比对布局与后端输出，差异逐条登记（不静默修正）。

勘察依据（transformers 5.23.0，2026-11-15）：
- `use_audio_in_video` 已废弃（v5.25 移除），语义由 `load_audio_from_video` 承接；
- 交错在 processor 层按 2 秒值域分块（position_id_per_seconds=25 × seconds_per_chunk=2.0
  = 50 位置单位/窗），块内视频块先于音频块；
- 视频 token 时间值 = int(帧下标 × temporal_patch_size/fps × pip)（.long() 截断），
  音频值 = 全局 token 下标（稠密，1 位置单位 ≈ 40ms = 1 个音频 token）——TMRoPE 对齐技巧；
- get_rope_index 的交错分支在模型侧按同一值域再次分块，与 processor 占位串一致。
"""

from __future__ import annotations

from typing import Any

import numpy as np

TIME_DEFAULTS = {
    "position_id_per_seconds": 25,
    "seconds_per_chunk": 2.0,
    "temporal_patch_size": 2,
    "audio_token_ms": 40.0,  # hop 160 @16kHz = 10ms/mel 帧；两个 stride-2 → 每 token ≈ 4 mel 帧
}


def layout_kwargs_for_backend(
    position_id_per_seconds: float = TIME_DEFAULTS["position_id_per_seconds"],
    seconds_per_chunk: float = TIME_DEFAULTS["seconds_per_chunk"],
) -> dict[str, Any]:
    """L1 布局参数 → 后端等价参数：块长取 seconds_per_chunk，块内视频先行（P0-01 勘察确认）。"""
    return {"video_block_s": seconds_per_chunk, "within_block_order": "video_first"}


def expected_media_runs(
    grid_t: int,
    n_audio_tokens: int,
    fps: float,
    tokens_per_frame: int = 1,
    position_id_per_seconds: float = TIME_DEFAULTS["position_id_per_seconds"],
    seconds_per_chunk: float = TIME_DEFAULTS["seconds_per_chunk"],
    temporal_patch_size: float = TIME_DEFAULTS["temporal_patch_size"],
) -> list[dict[str, Any]]:
    """按布局语义（值域分块、块内视频先行）推导后端期望的媒体 token 运行长度序列。

    grid_t 为后端 patch 网格的 t 维（temporal_patch_size=2 帧合 1 patch）；patch p 的位置值
    = int(p × temporal_patch_size/fps × pip)（.long() 截断）；音频 token 值 = 全局下标；
    窗口 = [j×(pip×spc), (j+1)×(pip×spc))；每窗先视频后音频，空侧跳过（与后端循环一致）。
    """
    window = int(position_id_per_seconds * seconds_per_chunk)
    step = temporal_patch_size / float(fps) * position_id_per_seconds
    video_values = [int(p * step) for p in range(grid_t)]
    n_windows = max(
        (video_values[-1] // window + 1) if video_values else 0,
        (n_audio_tokens + window - 1) // window if n_audio_tokens else 0,
    )
    runs: list[dict[str, Any]] = []
    for j in range(n_windows):
        n_video_frames = sum(1 for v in video_values if j * window <= v < (j + 1) * window)
        if n_video_frames:
            runs.append({"modality": "video", "n_tokens": n_video_frames * tokens_per_frame})
        na = min(n_audio_tokens, (j + 1) * window) - min(n_audio_tokens, j * window)
        if na:
            runs.append({"modality": "audio", "n_tokens": na})
    return runs


def backend_placeholder_string(
    processor,
    *,
    grid_t: int,
    tokens_per_frame: int,
    n_audio_tokens: int,
    fps: float,
    position_id_per_seconds: float = TIME_DEFAULTS["position_id_per_seconds"],
    seconds_per_chunk: float = TIME_DEFAULTS["seconds_per_chunk"],
    temporal_patch_size: float = TIME_DEFAULTS["temporal_patch_size"],
) -> str:
    """调用后端自身的 replace_video_with_audio_token 产出占位串（后端代码，非复刻）。

    构造的是该函数消费的最小输入表：video_grid_thw（patch 网格）、video_second_per_grid、
    num_audio_tokens、video_to_audio_indices。grid_h/grid_w 取 2×tokens_per_frame 使
    merge 后每帧恰好 tokens_per_frame 个 token。
    """
    import torch

    side = 2 * int(round(tokens_per_frame**0.5))
    if (side // 2) ** 2 != tokens_per_frame:
        raise ValueError(f"tokens_per_frame 必须是平方数以匹配 merge_size=2: {tokens_per_frame!r}")
    grid_h = grid_w = side
    video_inputs = {
        "video_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long),
        "video_second_per_grid": [temporal_patch_size / fps],
    }
    audio_inputs = {"num_audio_tokens": [n_audio_tokens]}
    return processor.replace_video_with_audio_token(
        video_inputs,
        video_idx=0,
        audio_inputs=audio_inputs,
        audio_idx=0,
        video_to_audio_indices=[0],
        load_audio_from_video=True,
        position_id_per_seconds=position_id_per_seconds,
        seconds_per_chunk=seconds_per_chunk,
    )


def parse_media_runs_from_string(placeholder: str, processor) -> dict[str, Any]:
    """占位串 → 媒体运行长度序列（借后端 tokenizer 把特殊 token 变 id 后解析）。"""
    tok = processor.tokenizer
    ids = tok(placeholder, add_special_tokens=False, return_tensors=None)["input_ids"]
    special = {
        "video": processor.video_token,
        "audio": processor.audio_token,
        "audio_bos": processor.audio_bos_token,
        "audio_eos": processor.audio_eos_token,
    }
    special_ids = {name: tok.convert_tokens_to_ids(s) for name, s in special.items()}
    runs: list[dict[str, Any]] = []
    bos_eos: list[str] = []
    cur_mod: Optional[str] = None
    cur_len = 0
    for t in ids:
        kind = next((k for k, v in special_ids.items() if t == v), None)
        if kind in ("audio_bos", "audio_eos"):
            if cur_mod is not None:
                runs.append({"modality": cur_mod, "n_tokens": cur_len})
            cur_mod, cur_len = None, 0
            bos_eos.append(kind)
        elif kind in ("video", "audio"):
            if cur_mod != kind:
                if cur_mod is not None:
                    runs.append({"modality": cur_mod, "n_tokens": cur_len})
                cur_mod, cur_len = kind, 1
            else:
                cur_len += 1
        else:
            if cur_mod is not None:
                runs.append({"modality": cur_mod, "n_tokens": cur_len})
            cur_mod, cur_len = None, 0
            runs.append({"modality": "text", "n_tokens": 1})
    if cur_mod is not None:
        runs.append({"modality": cur_mod, "n_tokens": cur_len})
    return {"media_runs": [r for r in runs if r["modality"] in ("video", "audio")], "all_runs": runs, "bos_eos": bos_eos}


def compare_media_runs(backend_runs: list[dict], expected_runs: list[dict]) -> dict[str, Any]:
    """后端运行序列 vs 布局期望。差异逐条登记，不做修正。"""
    diffs: list[str] = []
    n = max(len(backend_runs), len(expected_runs))
    for i in range(n):
        got = backend_runs[i] if i < len(backend_runs) else None
        exp = expected_runs[i] if i < len(expected_runs) else None
        if got != exp:
            diffs.append(f"run#{i}: 后端 {got} != 布局期望 {exp}")
    return {"matched": not diffs, "diffs": diffs, "backend_runs": backend_runs, "expected_runs": expected_runs}


SNAPSHOT_SCHEMA = "av_align.backend_snapshot/0-draft"  # v2 §3.2 后端快照（schema 待 G2 冻结）


def build_backend_snapshot(
    *,
    backend_version: str,
    model_id: str,
    grid_t: int,
    tokens_per_frame: int,
    n_audio_tokens: int,
    fps: float,
    video_unit_ids: list[str],
    audio_unit_ids: list[str],
    special_ids: dict[str, int],
    position_id_per_seconds: float = TIME_DEFAULTS["position_id_per_seconds"],
    seconds_per_chunk: float = TIME_DEFAULTS["seconds_per_chunk"],
    temporal_patch_size: float = TIME_DEFAULTS["temporal_patch_size"],
) -> dict[str, Any]:
    """构造 v2 §3.2 的后端快照记录（A 侧，packaging 层；前向入参快照待 P0-03/G2）。

    unit→token 映射与位置值由已验证的打包规则推导（tests 内双级核验支撑）：
    - 视频：第 k 个视频运行覆盖连续 patch；patch p → tokens_per_frame 个 token，
      位置值 = int(p × tps/fps × pip)（.long() 截断，显式记录）；
    - 音频：chunk k → 全局音频 token 下标 k，位置稠密（逐 run 连续 +1，段间偏移由
      rope 核验记录，不在本函数断言）。
    """
    step = temporal_patch_size / float(fps) * position_id_per_seconds
    window = int(position_id_per_seconds * seconds_per_chunk)
    video_values = [int(p * step) for p in range(grid_t)]
    n_windows = max(
        (video_values[-1] // window + 1) if video_values else 0,
        (n_audio_tokens + window - 1) // window if n_audio_tokens else 0,
    )
    runs: list[dict[str, Any]] = []
    token_idx = 0
    patch_cursor = 0
    video_unit_token_map: dict[str, dict[str, Any]] = {}
    for j in range(n_windows):
        n_patches = sum(1 for v in video_values if j * window <= v < (j + 1) * window)
        if n_patches:
            runs.append({"modality": "video", "n_tokens": n_patches * tokens_per_frame, "window": j})
            for i in range(n_patches):
                p = patch_cursor + i
                uid = video_unit_ids[p]
                video_unit_token_map[uid] = {
                    "token_range": [token_idx + i * tokens_per_frame, token_idx + (i + 1) * tokens_per_frame],
                    "position_value": video_values[p],
                    "truncation": "int(p*step) 截断，step = tps/fps*pip = %s" % step,
                }
            token_idx += n_patches * tokens_per_frame
            patch_cursor += n_patches
        na_lo = min(n_audio_tokens, j * window)
        na_hi = min(n_audio_tokens, (j + 1) * window)
        if na_hi > na_lo:
            runs.append({"modality": "audio", "n_tokens": na_hi - na_lo, "window": j})
            token_idx += na_hi - na_lo

    audio_unit_token_map = {
        uid: {"token_index": k, "position_rule": "dense arange（段间偏移见 rope 核验）"}
        for k, uid in enumerate(audio_unit_ids)
    }
    return {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "backend": f"transformers=={backend_version}",
        "model_id": model_id,
        "weights_sha": None,  # 待 P0-03/G2 固定权重后填（v2 复现标识要求）
        "time_constants": {
            "position_id_per_seconds": position_id_per_seconds,
            "seconds_per_chunk": seconds_per_chunk,
            "temporal_patch_size": temporal_patch_size,
            "audio_token_ms": TIME_DEFAULTS["audio_token_ms"],
            "tokens_per_chunk": int(position_id_per_seconds * seconds_per_chunk),
        },
        "special_token_ids": special_ids,
        "media_runs": runs,
        "unit_token_map": {"video": video_unit_token_map, "audio": audio_unit_token_map},
        "unverified_fields": [
            "weights_sha（权重未加载）",
            "前向入参快照（input_ids/掩码/媒体张量指纹/选项 logits，待 G2 原生回放）",
            "音频段间位置偏移的绝对值（rope 核验为差分口径）",
        ],
    }


def tiny_thinker(device: str = "cpu"):
    """tiny-config 实例化 thinker 基类：仅调用 get_rope_index（与权重无关，随机初始化即可）。"""
    import torch
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
        Qwen2_5OmniTextConfig,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniVisionEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    )

    config = Qwen2_5OmniThinkerConfig(
        audio_config=Qwen2_5OmniAudioEncoderConfig(
            num_mel_bins=128, encoder_layers=1, encoder_attention_heads=4,
            encoder_ffn_dim=64, d_model=32, output_dim=32, n_window=4,
        ),
        vision_config=Qwen2_5OmniVisionEncoderConfig(
            depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
            out_hidden_size=32, window_size=8, fullatt_block_indexes=[0],
            spatial_merge_size=2, temporal_patch_size=2, patch_size=14,
        ),
        text_config=Qwen2_5OmniTextConfig(
            vocab_size=152064, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=8192,
        ),
    )
    model = Qwen2_5OmniPreTrainedModelForConditionalGeneration(config)
    # get_rope_index 以模型属性读取合并尺寸（正式加载路径由 post_init/加载器设置）；
    # vision_start_token_id 在 omni 顶层 config，thinker 级缺省，按 tokenizer <|vision_bos|> 补
    model.spatial_merge_size = config.vision_config.spatial_merge_size
    config.vision_start_token_id = 151652
    model.eval()
    return model.to(device)


def rope_temporal_positions(
    model,
    input_ids,
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    audio_mel_len: int,
    fps: float,
    use_audio_in_video: bool = True,
    position_id_per_seconds: float = TIME_DEFAULTS["position_id_per_seconds"],
    seconds_per_chunk: float = TIME_DEFAULTS["seconds_per_chunk"],
    temporal_patch_size: float = TIME_DEFAULTS["temporal_patch_size"],
) -> np.ndarray:
    """调用后端 get_rope_index，返回时间维 position_ids（numpy，1D）。"""
    import torch

    # 勘察怪癖（5.23.0）：get_rope_index 的 audio_seqlens 虽标注 LongTensor，实现里
    # torch.arange(标量) 要求纯数字，须传 Python 列表；second_per_grids 走 .cpu().float()，传张量。
    with torch.no_grad():
        position_ids, _ = model.get_rope_index(
            input_ids,
            None,
            torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long),
            torch.ones_like(input_ids),
            use_audio_in_video,
            [int(audio_mel_len)],
            torch.tensor([temporal_patch_size / fps], dtype=torch.float),
        )
    arr = np.asarray(position_ids.detach().cpu().numpy())
    if arr.ndim == 3:  # (3, bs, L)：时间维在轴 0；如为 (bs, 3, L) 需在轴 1 取时间
        if arr.shape[0] != 3 and arr.shape[1] == 3:
            arr = arr[:, 0, :]
        else:
            arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] == 3:
        arr = arr[0]
    return arr.reshape(-1)


def rope_report(
    positions: np.ndarray,
    token_ids: list[int],
    special_ids: dict[str, int],
    *,
    fps: float,
    tokens_per_frame: int,
    position_id_per_seconds: float = TIME_DEFAULTS["position_id_per_seconds"],
    seconds_per_chunk: float = TIME_DEFAULTS["seconds_per_chunk"],
    temporal_patch_size: float = TIME_DEFAULTS["temporal_patch_size"],
) -> dict[str, Any]:
    """对 get_rope_index 的时间维位置做逐运行核验（offset 无关，直接扫 input_ids）：

    - 视频运行：帧内 token 同值；帧值 = int(全局帧下标 × tps/fps × pip)（.long() 截断）；
    - 音频运行：逐 token +1（稠密，TMRoPE 对齐假设）；
    - 文本/bos/eos 位置不核验（后端内部约定，登记为观测边界）。
    """
    step = temporal_patch_size / float(fps) * position_id_per_seconds
    diffs: list[str] = []
    video_frame_cursor = 0
    n = len(token_ids)
    i = 0
    while i < n:
        t = token_ids[i]
        if t == special_ids.get("video"):
            j = i
            while j < n and token_ids[j] == special_ids["video"]:
                j += 1
            seg = positions[i:j]
            nf = (j - i) // tokens_per_frame
            expected: list[int] = []
            for f in range(video_frame_cursor, video_frame_cursor + nf):
                expected.extend([int(f * step)] * tokens_per_frame)
            got = np.diff(seg).tolist()
            exp = np.diff(np.array(expected)).tolist()
            if got != exp:
                diffs.append(f"video run@{i}: 帧值差分 后端{got[:8]} != 期望{exp[:8]}")
            video_frame_cursor += nf
            i = j
        elif t == special_ids.get("audio"):
            j = i
            while j < n and token_ids[j] == special_ids["audio"]:
                j += 1
            seg = positions[i:j]
            if not np.all(np.diff(seg) == 1):
                diffs.append(f"audio run@{i}: 非稠密(+1)，实测差分 {np.diff(seg)[:8].tolist()}")
            i = j
        else:
            i += 1
    return {"matched": not diffs, "diffs": diffs}
