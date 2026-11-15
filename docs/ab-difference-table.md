# A/B 差异表（G2 交付物底稿）

依据：研发方案 v2 §4.2（"A/B 差异表记录输入与配置哈希、字段、两侧实测值、预期值或容差、复现命令和定位阶段"；"每条对照标记一致、差异或未验证，并注明证据来源"）。

- **A 侧**已实测：Transformers，两个版本（4.57.6 @ Colab bhqc20 A100，2026-11-15；5.23.0 @ 本机勘察）。证据：`artifacts/音画时间机制/p0-01-colab-证据/20261115/`（report.json / verification.json）与 `P0-01-transformers-勘察记录.md`。
- **B 侧**未验证：mlx-vlm / qwen3_omni_moe，待 Mac 执行 `p2-01-mac-运行包/`（16 GiB 内存超限实测见 P0-02 记录；≥24 GiB 机型或低位宽工件就位后回填）。
- 结论标记口径：**一致 / 差异 / 未验证**；"未验证"不进发布主张。

## 差异表

| # | 条目 | A 实测（4.57.6，Colab） | A 实测（5.23.0，本机） | B 实测（待 P2-01） | 预期/容差 | 结论 | 证据来源 | 定位阶段 |
|---|---|---|---|---|---|---|---|---|
| 1 | position_id_per_seconds | 25 | 25 | 13（config 实测待复核） | P0-02 config 读取值 | **代际级差异**（A/B 家族不同代） | [P0-01 记录](../../artifacts/音画时间机制/P0-01-transformers-勘察记录.md) · [P0-02 记录](../../artifacts/音画时间机制/P0-02-mlxvlm-勘察记录.md) | G0 |
| 2 | seconds_per_chunk | 2.0 | 2.0 | 2.0（config 待复核） | 2.0 | 预期一致（待 B 复核） | 同上 | G0 |
| 3 | 交错发生层 | processor `replace_video_with_audio_token` | 同（改名 `load_audio_from_video` 路径） | processor 层（同构，待复核） | 两侧均在 processor 层 | 预期一致（待 B） | P0-01/P0-02 | G0 |
| 4 | 音频时间值 | 全局 token 下标（稠密 arange，1 单位≈40ms） | 同 | arange（待复核） | 稠密 | 预期一致（待 B） | P0-01 §2 | G0 |
| 5 | 视频 token 时间值 | int(帧/patch 下标 × tps/fps × 25)，.long() 截断 | 同 | int(p × tps/fps × 13)（待复核） | pip 差异由换算吸收 | 代际级差异 | P0-01 §5 | G0 |
| 6 | grid t 维粒度 | patch（temporal_patch_size=2 帧/patch） | 同 | 待复核（mlx video processor merge 参数） | patch 粒度 | 未验证 | P0-01 勘察修正 | G0 |
| 7 | audio_seqlens 语义 | 有效 mel 帧数（运行时实测 602；内部 None 时推导待源码核对） | 纯数字列表（Tensor 抛 TypeError 的怪癖） | omni_utils #1620 手工换算路径（待复核） | 换算规则按侧登记 | **实现级差异**（A 侧怪癖已登记） | report.json alignment_matrix · omni_utils L61–80 | G2 |
| 8 | use_audio_in_video 音频来源 | 调用方单独传入（librosa 抽容器音轨） | processor 内化（load_audio_from_video） | conversation audio part（待复核） | 按侧登记 | **实现级差异** | P0-01 §3 / P0-02 | G2 |
| 9 | 原生默认分支 | use_audio_in_video=False（inputs 不带键时） | use_audio_in_video 已废弃 | 待复核（默认值） | 按侧登记 | 未验证 | report.json alignment_matrix | G2 |
| 10 | 接口语义敏感性 | flag True/False 改变答案（D→B，选项 logits 实测） | 同路径待测 | 待测（单因子对照） | —— | 未验证（A 侧已证敏感） | report.json §alignment | G3（研究线引用） |
| 11 | 原生两次前向离散入参一致 | 位级（jitter=0） | 位级（jitter=0，本机可复核） | 待测 | 位级 | A 通过；B 未验证 | report.json | G2-1 |
| 12 | 显式回放等价 | 位级（diff=0 ≤ 冻结容差 0） | 待本机全前向（可做） | 待测 | ≤ 冻结抖动 | A 通过；B 未验证 | report.json | G2-1 |
| 13 | 位置干预（只动 position_ids） | passthrough 重述 0.0；单路 +1s 扰动 0.172；位移 [25] 有效；共同平移 0.3125 | 同路径可复核 | **只读重述**（媒体组织层，不做 logits 断言——v2 §4.2-3 口径） | 方向/幅度布局层一致 | A 通过；B 未验证 | tensors_e1_fixed.npz 独立复核 | G2-2/3 |
| 14 | 峰值内存 | 10.06 GiB（A100，bf16 3B） | —— | 待测（ru_maxrss；16 GiB 机型预期超限） | 预算比对 | B 未验证 | report.json | G2-3 |

## 复现命令（A 侧）

```bash
# Colab（4.57.6）：运行包 av_align_colab_p0_01.ipynb（或 CLI 通道逐格，见 p0-01-colab-证据/20261115/executed_cells/）
# 本机（5.23.0）：.venv && pytest tests/test_backend_transformers.py（packaging/rope 级）
```

B 侧回填规则：Mac 运行包执行后，本表 B 列逐条回填实测值并把"未验证"改为一致/差异；任何差异条目按 v2 §4.2 流程登记根因与影响（实现级 vs 代际级分栏），可复现缺陷走 MLX-VLM 回流。
