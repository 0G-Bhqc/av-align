# 外部依赖许可审计（G4 发布门）

审计日期 2026-11-15。取证方式：本机安装环境的 dist-info `License-Expression`/分类器与随附许可文件（非凭记忆）；审计对象为 av_align 0.2.0 声明的依赖及其传递依赖。结论先行：**全部为宽松许可或文件级弱 copyleft，允许以 Apache-2.0 发布 av_align 并分发依赖清单；无需在发行目录内捆绑第三方许可正文（保留随附许可文件即可），但发行时须附本表。**

## 运行时依赖（pyproject `dependencies`）

| 包 | 版本（实测） | 许可（dist-info 取证） | 审计结论 |
|---|---|---|---|
| numpy | 2.4.6 | BSD-3-Clause（含捆绑的 0BSD/MIT/Zlib/CC0-1.0 组件，见随附 licenses/） | ✅ 宽松；捆绑组件均为宽松许可，随附文件已覆盖 |

## 可选 extras

| extra | 包 | 版本 | 许可 | 结论 |
|---|---|---|---|---|
| test | pytest | 9.2.0 | MIT | ✅（传递：pluggy MIT、iniconfig MIT、pygments BSD-2-Clause） |
| transformers | transformers | 5.23.0（Colab 实测另含 4.57.6） | Apache-2.0 | ✅ |
| （transformers 传递） | torch | 2.15.0+cpu（Colab：2.14.1+cu130） | Apache-2.0 + Apache-2.0 WITH LLVM-exception + BSD-2/3-Clause 组合（License-Expression，2.15 起） | ✅ 宽松组合 |
| | torchvision | 0.29.2 | BSD-3 | ✅ |
| | pillow | 13.0.0 | MIT-CMU | ✅ |
| | huggingface_hub | 2.4.0 | Apache-2.0 | ✅ |
| | tokenizers | 0.23.3 | Apache-2.0（分类器；官方仓库 Apache-2.0） | ✅ |
| | safetensors | 0.8.0 | Apache-2.0 | ✅ |
| （其余传递，抽样全列） | filelock / fsspec / packaging / typing_extensions / networkx / jinja2 / MarkupSafe / idna / regex / mpmath / PyYAML / tqdm / sympy | — | MIT / BSD-3 / Apache-2.0 OR BSD-2 / PSF-2.0 / BSD-3 / BSD / BSD-3 / BSD-3 / Apache-2.0 AND CNRI-Python / BSD / MIT / MPL-2.0 AND MIT / BSD | ✅ tqdm 为 MPL-2.0（文件级弱 copyleft，独立分发不受传染）；regex 双许可宽松 |

## 非包依赖（发布说明须登记）

| 项 | 许可 | 备注 |
|---|---|---|
| Qwen/Qwen2.5-Omni-3B 权重 | 按官方模型卡（Qwen 系列许可，非 Apache） | 权重不随 av_align 分发；引用实验须注明 revision `f75b40e3…` 与模型卡条款 |
| Colab 会话内使用 librosa | ISC | 仅 P0-01 执行通道内抽容器音轨用，非 av_align 依赖 |
| ffmpeg（Colab 预装） | LGPL/GPL 视构建 | 仅容器 mux 用，非 av_align 依赖；发行示例若含 mux 步骤须单独登记构建来源 |

## 发行纪律（沿方案 §6 与 pfdenoise 先例）

1. 发行目录只包含许可允许的内容：av_align 自身代码（Apache-2.0）+ 本表 + 各依赖版本清单；**不捆绑**任何第三方许可正文副本（pip 安装自带随附许可文件）。
2. 权重、数据、媒体样例不进发行目录；合成事件夹具代码为本仓库自有（Apache-2.0）。
3. 公开发布前复跑本审计（依赖版本可能变化）；每次变更依赖须在本表追加一行并注明取证方式。
