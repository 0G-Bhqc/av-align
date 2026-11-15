# 发布范围与限制表（G4，2026-11-15）

按方案 §5"按通过的能力范围发布"。本表是发布说明的底稿；每次能力变更须更新。

## 已验证能力（可发布主张）

| 能力 | 证据 |
|---|---|
| L1 布局核心（av_align.layout/1：ns 时间、单元 ID + 时间来源、源/目标分离、四干预、配对表、内容哈希） | 干净环境 20/20 测试（clean-install-verification.md） |
| 后端 A 打包与位置编码的只读重述 | 双级核验（packaging 运行序列一致；tiny-config get_rope_index 逐运行一致） |
| 后端 A 全前向通道：原生/显式回放等价 + 位置干预 | Colab bhqc20 A100 实测 A1–A6 全过（transformers 4.57.6，Qwen2.5-Omni-3B @ f75b40e3…） |
| §4.2-2 位置干预语义 | passthrough 位级重述 0.0；单路 +1s 扰动 0.172、共同平移 0.3125、argmax 不变、位移 [25] 有效 |

## 未验证 / 未包含（不可作主张）

| 项 | 状态 | 解锁条件 |
|---|---|---|
| 后端 B（mlx-vlm / qwen3_omni_moe） | 代码就绪未执行（Windows 无 MLX） | Apple Silicon 执行 `verify_on_metal`（P2-01） |
| 后端 C（跨家族复现） | 候选清单已立、决议未冻结 | P0-03/P2-01 后按标准打分冻结（docs/backend-c-candidates.md） |
| P0-03 错例回放 | 需研究线首个确认错例 | 研究线首轮交付错例后补同格式快照 |
| G2 的 A/B 差异表 | 依赖 B 侧 | P2-01 |
| G4 完整发布（PyPI/远端仓库/安装验证多平台） | 本仓库为本地私有，未公开 | 命名决定（方案 §1）+ 远端仓库建立 |

## 已知缺陷与怪癖（随发布披露）

| # | 内容 | 状态 |
|---|---|---|
| E1 | 证据 npz 中 pos_native（True 分支基座）与 pos_single/pos_both（False 分支基座）基底混用，npz 不能独立复核位移精确性；A4 以运行时记录（位移 [25]）佐证 | **P0-03 开启 Colab 会话时的首个动作**：同基座重存四组张量（预案脚本 `artifacts/音画时间机制/p0-01-colab-证据/20261115/E1-remediate.py`） |
| Q1 | transformers 4.57.6 原生默认 `use_audio_in_video=False` 分支；官方 docstring 用法（=True）改变答案（D→B） | 已实测登记（report.json alignment_matrix）——接口语义敏感性，交研究线引用 |
| Q2 | False 分支对 audio_seqlens 候选值（600/300/150）均位级一致——交错 input_ids 下位置赋值对音频长度不敏感 | 已实测登记（怪癖） |
| Q3 | 5.23 已废弃 `use_audio_in_video`（v5.25 移除）；适配器须按版本分支（`load_audio_from_video` / `audio_seqlens` 纯数字列表） | 双路径兼容已实现并记录（P0-01 勘察记录 §3） |

## 通道差异登记

- Colab 执行走 CLI 通道（`colab exec` 逐格 + `download` 回传证据），证据落 `p0-01-colab-证据/20261115/`；网页通道（notebook 直跑）为并行路径，验收口径相同。运行包 README 已登记网页通道执行前须同步的三条修复（apply_chat_template、音频单独传入、按同分支计算显式位置）。
- 干净环境验证发现并修复 F1/F2 两条发布元数据缺陷（clean-install-verification.md）。
