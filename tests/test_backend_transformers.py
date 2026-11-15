"""后端 A（Transformers / Qwen2.5-Omni）CPU 级只读重述验证（P1-02）。

- 不下载模型权重：packaging 用 processor 自身的 replace_video_with_audio_token（后端代码），
  位置用 tiny-config 实例化 thinker 基类后调用其 get_rope_index（与权重无关）。
- 勘察修正（2026-11-15）：后端 video_grid_thw 的 t 维以 patch 计（temporal_patch_size=2
  帧合 1 patch）；L1 对齐粒度取 patch：fps_eff = fps / temporal_patch_size。
- 需要 HF 网络（processor 配置文件，数 MB）。设 AV_ALIGN_OFFLINE=1 可跳过。
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("AV_ALIGN_OFFLINE") == "1":
    pytest.skip("AV_ALIGN_OFFLINE=1：跳过后端网络验证", allow_module_level=True)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from av_align.adapters import transformers_adapter as ta  # noqa: E402

PROC_ID = "Qwen/Qwen2.5-Omni-3B"

# 勘察组合：6 patch（=12 原始帧 @4fps，3 s）、grid 4×4（merge 后每 patch 4 token）、
# 75 个音频 token（3 s @40ms）、audio mel 长度 301（经双 stride-2 后 ≈ 75）
P = dict(grid_t=6, tokens_per_frame=4, n_audio_tokens=75, fps=4.0, audio_mel_len=301)

EXPECTED_RUNS = [
    {"modality": "video", "n_tokens": 16},  # 窗 0（0–2 s）：4 patch × 4 token
    {"modality": "audio", "n_tokens": 50},
    {"modality": "video", "n_tokens": 8},  # 窗 1（2–3 s）：2 patch × 4 token
    {"modality": "audio", "n_tokens": 25},
]


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(PROC_ID)


def _special_ids(processor):
    tok = processor.tokenizer
    return {
        "video": tok.convert_tokens_to_ids(processor.video_token),
        "audio": tok.convert_tokens_to_ids(processor.audio_token),
        "audio_bos": tok.convert_tokens_to_ids(processor.audio_bos_token),
        "audio_eos": tok.convert_tokens_to_ids(processor.audio_eos_token),
        "vision_bos": tok.convert_tokens_to_ids(processor.vision_bos_token)
        if hasattr(processor, "vision_bos_token")
        else None,
    }


def test_backend_time_constants():
    assert ta.TIME_DEFAULTS["position_id_per_seconds"] == 25
    assert ta.TIME_DEFAULTS["seconds_per_chunk"] == 2.0
    assert ta.layout_kwargs_for_backend() == {"video_block_s": 2.0, "within_block_order": "video_first"}


def test_backend_packing_matches_layout(processor):
    """后端自身的占位串构造 vs 布局期望（值域分块、块内视频先行）。"""
    s = ta.backend_placeholder_string(
        processor, **{k: v for k, v in P.items() if k in ("grid_t", "tokens_per_frame", "n_audio_tokens", "fps")}
    )
    parsed = ta.parse_media_runs_from_string(s, processor)
    expected = ta.expected_media_runs(
        P["grid_t"], P["n_audio_tokens"], fps=P["fps"], tokens_per_frame=P["tokens_per_frame"]
    )
    report = ta.compare_media_runs(parsed["media_runs"], expected)
    assert report["matched"], report["diffs"]
    assert parsed["media_runs"] == EXPECTED_RUNS
    assert parsed["bos_eos"] == ["audio_bos", "audio_eos"]


def test_layout_block_mode_matches_expected_runs():
    """同一勘察组合在 L1 侧（build_layout + 后端等价参数）产出一致运行序列。

    L1 以 patch 为视频槽位（fps_eff = fps / temporal_patch_size = 2 patch/s）。
    """
    from av_align import AudioTrack, VideoTrack, build_layout

    video = VideoTrack.from_fps(n_frames=P["grid_t"], fps=P["fps"] / 2.0)  # patch 槽位
    audio = AudioTrack.from_chunk_times([(i * 0.04, (i + 1) * 0.04) for i in range(P["n_audio_tokens"])])
    plan = build_layout(video, audio, **ta.layout_kwargs_for_backend())
    seq = plan.interleaving["sequence"]
    runs = []
    for s in seq:
        n = P["tokens_per_frame"] if s["modality"] == "video" else 1  # L1 槽位 = patch，后端 4 token/patch
        if runs and runs[-1]["modality"] == s["modality"]:
            runs[-1]["n_tokens"] += n
        else:
            runs.append({"modality": s["modality"], "n_tokens": n})
    expected = ta.expected_media_runs(
        P["grid_t"], P["n_audio_tokens"], fps=P["fps"], tokens_per_frame=P["tokens_per_frame"]
    )
    assert runs == expected


def test_rope_positions_match_layout(processor):
    """tiny thinker 的 get_rope_index：视频 patch 值（截断）、音频稠密性，逐运行核验。"""
    model = ta.tiny_thinker()
    tok = processor.tokenizer
    s = ta.backend_placeholder_string(
        processor, **{k: v for k, v in P.items() if k in ("grid_t", "tokens_per_frame", "n_audio_tokens", "fps")}
    )
    special = _special_ids(processor)
    # 官方输入结构：…text, <|vision_bos|>, [audio_bos, 媒体占位…, audio_eos], text…
    # get_rope_index 的 text_len - 2 假设媒体前有两个起始 token（vision_bos + audio_bos）
    vision_bos_id = tok.convert_tokens_to_ids("<|vision_bos|>")
    assert vision_bos_id is not None
    prefix = [vision_bos_id]
    text_ids = tok("demo", add_special_tokens=False)["input_ids"]
    body_ids = tok(s, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([text_ids + prefix + body_ids + text_ids], dtype=torch.long)

    positions = ta.rope_temporal_positions(
        model,
        input_ids,
        grid_t=P["grid_t"],
        grid_h=4,
        grid_w=4,
        audio_mel_len=P["audio_mel_len"],
        fps=P["fps"],
    )
    assert positions.shape[0] == input_ids.shape[1], (positions.shape, input_ids.shape)
    report = ta.rope_report(
        positions,
        input_ids[0].tolist(),
        {k: special[k] for k in ("video", "audio")},
        fps=P["fps"],
        tokens_per_frame=P["tokens_per_frame"],
    )
    assert report["matched"], report["diffs"]
