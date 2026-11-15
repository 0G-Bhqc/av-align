"""后端 B 适配器：mlx-vlm / qwen3_omni_moe（Apple Silicon 专用）。

⚠️ 状态（2026-11-15）：代码就绪，**Metal 执行待 Apple Silicon**（Windows 勘察机无法运行 MLX）。
执行步骤见 verify_on_metal() docstring。

勘察依据（P0-02，mlx-vlm 0.7.12 源码级；对研发方案 v2 §1 B 轨的确认证据）：
- **0.7.12 无 qwen2_5_omni**——v2 §1 表中"Qwen2.5-Omni-3B 在 mlx-vlm 上的联合音画入口
  仍待验证"已可回答：当前版本不存在该入口；社区 4-bit 工件（giangndm/qwen2.5-omni-3b-mlx-4bit）
  走 mlx-lm，与 v2 的判断一致。B 轨两条路径：(a) qwen3_omni_moe（本适配器），(b) mlx-lm 工件另查；
- qwen3_omni_moe 交错在 processor 层：video token 时间值 = 帧下标 × video_second_per_grid × 13
  （平局视频先行），音频 = arange；按值域分块（seconds_per_chunk=2.0 → 26 位置单位/窗），
  与后端 A 同构、时间分辨率不同；
- 入口 `omni_utils.prepare_omni_inputs(processor, conversation, use_audio_in_video)`；
- 注意 #1620 修复：feature_attention_mask（样本域）→ audio_feature_lengths（mel 帧域）需换算。

对照语义（v2 §4.2-4）：A/B 为同家族两代实现对照；"实现级差异"与"模型代际差异"分开表述。
C（跨家族复现）候选见 docs/backend-c-candidates.md。
"""

from __future__ import annotations

from typing import Any

TIME_DEFAULTS = {
    "position_id_per_seconds": 13,  # mlx qwen3_omni_moe config 默认（后端 A 为 25）
    "seconds_per_chunk": 2.0,
    "audio_token_ms": 40.0,
}


def layout_kwargs_for_backend(
    position_id_per_seconds: float = TIME_DEFAULTS["position_id_per_seconds"],
    seconds_per_chunk: float = TIME_DEFAULTS["seconds_per_chunk"],
) -> dict[str, Any]:
    """L1 布局参数 → 后端 B 等价参数。与 A 同构：2 秒值域分块、块内视频先行。"""
    return {"video_block_s": seconds_per_chunk, "within_block_order": "video_first"}


def verify_on_metal(model_id: str = "mlx-community/Qwen3-Omni-3B-4bit"):
    """在 Apple Silicon 上执行 A/B 对照（P2-01 的执行体，本机无法运行）。

    步骤：
    1. `pip install mlx-vlm==0.7.12`，加载 `model_id`（16 GiB 建议 4-bit 档）；
    2. 用 av_align 合成事件夹具（fixtures.make_synthetic_event_fixture）构造输入媒体；
    3. 经 `omni_utils.prepare_omni_inputs(..., use_audio_in_video=True)` 得到占位与输入；
    4. 解析占位串媒体运行序列（复用 transformers_adapter.parse_media_runs 的解析逻辑，
       token 字符串以 mlx 模型 config 的 special tokens 为准）；
    5. 与 L1 build_layout(**layout_kwargs_for_backend()) 的运行序列比对，差异逐条登记；
    6. 干预只读重述：ReindexSide 在 conversation 媒体组织上的等价操作（改帧时间戳），
       记录干预前后 processor 输出的时间元数据（不做 logits 位断言）。

    产出：A/B 对照表（写入实验归档）+ 差异登记表。
    """
    raise NotImplementedError(
        "P2-01 待 Apple Silicon 执行：mlx 与 Metal 运行时在 Windows 上不可用。"
        "执行步骤见本函数 docstring；版本表与勘察记录见工作区 artifacts/音画时间机制/P0-02-mlxvlm-勘察记录.md"
    )
