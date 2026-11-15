# A/B 差异表（G2 交付物底稿）

依据：研发方案 v2 §4.2（"A/B 差异表记录输入与配置哈希、字段、两侧实测值、预期值或容差、复现命令和定位阶段"；"每条对照标记一致、差异或未验证，并注明证据来源"）。

- **A 侧**已实测：Transformers，两个版本（4.57.6 @ Colab bhqc20 A100，2026-11-15；5.23.0 @ 本机勘察）。证据：`artifacts/音画时间机制/p0-01-colab-证据/20261115/`（report.json / verification.json）与 `P0-01-transformers-勘察记录.md`。
- **B 侧打包层已实测（2026-11-15）**：GitHub Actions macos-latest（macOS-26.7.2 **arm64**，MLX GPU 设备，真 Metal 环境；mlx_vlm 0.7.12 / transformers 5.23.0，processor-only 无权重）。证据：`artifacts/音画时间机制/p2-01-证据/20261115/`（manifest 相符）+ Actions run 43708732004。vendored L1 SHA 跨平台复现一致（`5e988284…`）。B 侧 **全前向与内存项仍未验证**（实体 Mac 或 ≥24 GiB 机型）。
- 结论标记口径：**一致 / 差异 / 未验证**；"未验证"不进发布主张。

## 差异表

| # | 条目 | A 实测（4.57.6，Colab） | A 实测（5.23.0，本机） | B 实测（待 P2-01） | 预期/容差 | 结论 | 证据来源 | 定位阶段 |
|---|---|---|---|---|---|---|---|---|
| 1 | position_id_per_seconds | 25 | 25 | 13（**config 实测确认**） | P0-02 config 读取值 | **代际级差异**（A/B 家族不同代） | [P0-01 记录](../../artifacts/音画时间机制/P0-01-transformers-勘察记录.md) · [P0-02 记录](../../artifacts/音画时间机制/P0-02-mlxvlm-勘察记录.md) | G0 |
| 2 | seconds_per_chunk | 2.0 | 2.0 | 2.0（config 待复核） | 2.0 | 预期一致（待 B 复核） | 同上 | G0 |
| 3 | 交错发生层 | processor `replace_video_with_audio_token` | 同（改名 `load_audio_from_video` 路径） | processor 层 | 两侧均在 processor 层 | **实测差异**：B 侧不交错（见 #13） | P0-01/P0-02 | G0 |
| 4 | 音频时间值 | 全局 token 下标（稠密 arange，1 单位≈40ms） | 同 | arange（稠密，实测） | 稠密 | **实测一致**（B 侧音频 token ≈13/s=1 单位/token，代际分辨率差异） | P0-01 §2 | G0 |
| 5 | 视频 token 时间值 | int(帧/patch 下标 × tps/fps × 25)，.long() 截断 | 同 | int(p × tps/fps × 13)（公式同构，实测 pip=13） | pip 差异由换算吸收 | **代际级差异（实测确认）** | P0-01 §5 | G0 |
| 6 | grid t 维粒度 | patch（temporal_patch_size=2 帧/patch） | 同 | 待复核（mlx video processor merge 参数） | patch 粒度 | 未验证 | P0-01 勘察修正 | G0 |
| 7 | audio_seqlens 语义 | 有效 mel 帧数（运行时实测 602；内部 None 时推导待源码核对） | 纯数字列表（Tensor 抛 TypeError 的怪癖） | omni_utils #1620 手工换算路径（待复核） | 换算规则按侧登记 | **实现级差异**（A 侧怪癖已登记） | report.json alignment_matrix · omni_utils L61–80 | G2 |
| 8 | use_audio_in_video 音频来源 | 调用方单独传入（librosa 抽容器音轨） | processor 内化（load_audio_from_video） | conversation audio part（待复核） | 按侧登记 | **实现级差异** | P0-01 §3 / P0-02 | G2 |
| 9 | 原生默认分支 | use_audio_in_video=False（inputs 不带键时） | use_audio_in_video 已废弃 | 待复核（默认值） | 按侧登记 | 未验证 | report.json alignment_matrix | G2 |
| 10 | 接口语义敏感性 | flag True/False 改变答案（D→B，选项 logits 实测） | 同路径待测 | 待测（单因子对照） | —— | 未验证（A 侧已证敏感） | report.json §alignment | G3（研究线引用） |
| 11 | 原生两次前向离散入参一致 | 位级（jitter=0） | 位级（jitter=0，本机可复核） | 待测 | 位级 | A 通过；B 未验证 | report.json | G2-1 |
| 12 | 显式回放等价 | 位级（diff=0 ≤ 冻结容差 0） | 待本机全前向（可做） | 待测 | ≤ 冻结抖动 | A 通过；B 未验证 | report.json | G2-1 |
| 13 | 位置干预（只动 position_ids） | passthrough 重述 0.0；单路 +1s 扰动 0.172；位移 [25] 有效；共同平移 0.3125 | 同路径可复核 | **只读重述**（媒体组织层，不做 logits 断言——v2 §4.2-3 口径） | 方向/幅度布局层一致 | A 通过；B 未验证 | tensors_e1_fixed.npz 独立复核 | G2-2/3 |
| 14 | 峰值内存 | 10.06 GiB（A100，bf16 3B） | —— | 待测（ru_maxrss；16 GiB 机型预期超限，4bit=20.33 GiB 实测超） | 预算比对 | B 未验证 | report.json | G2-3 |
| 15 | **打包对照（use_audio_in_video=True + 独立 audio part）** | [V16 A26 V8 A25]（2s 值域分块交错，块内视频先行） | 同构（5.23 路径名不同） | **[V48 A78]（不交错：视频整体在前、音频整体在后）** | 两侧按各自 pip 值域分块 | **实现级差异（实测）**——B 侧该组合不交错，A/B 结论表述须分开 | p2-01 证据（Actions run 43708732004） | G2-3 |
| 16 | 音频 token 率（时间分辨率） | 25 token/s（40 ms/token） | 同 | **≈13 token/s（77 ms/token），与 pip=13 自洽** | 代际参数 | **代际级差异（实测确认）** | p2-01 证据 | G0 |

## 复现命令（A 侧）

```bash
# Colab（4.57.6）：运行包 av_align_colab_p0_01.ipynb（或 CLI 通道逐格，见 p0-01-colab-证据/20261115/executed_cells/）
# 本机（5.23.0）：.venv && pytest tests/test_backend_transformers.py（packaging/rope 级）
```

B 侧回填规则：Mac 运行包执行后，本表 B 列逐条回填实测值并把"未验证"改为一致/差异；任何差异条目按 v2 §4.2 流程登记根因与影响（实现级 vs 代际级分栏），可复现缺陷走 MLX-VLM 回流。
