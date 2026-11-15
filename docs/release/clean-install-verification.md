# 干净环境安装验证（G4 发布门）

验证日期 2026-11-15。方法：全新 venv（Python 3.11.15）+ **非 editable 安装** `av-align[test,transformers]`，从本地仓库构建 wheel 后安装，在仓库目录外运行全部 20 项测试（确保被测的是 site-packages 里的安装产物，不是源码树）。

## 结果

| 项 | 值 |
|---|---|
| venv | `D:\DevEnvironment\temp\ai-data\av_align-clean-20261115\.venv`（验证后可删） |
| 安装形态 | 非 editable，`uv pip install "D:/Develop/av_align[test,transformers]"` |
| 安装版本 | av-align **0.2.0**（`import av_align; av_align.__version__` 复核） |
| 测试结果 | **20 passed**（16 L1/性质/契约 + 4 后端 A，后者经 HF 网络加载 processor 配置，无权重） |
| Python | 3.11.15 |
| 关键包 | numpy 2.4.6 / pytest 9.2.0 / torch 2.15.0+cpu / transformers 5.23.0 / pillow 13.0.0 / torchvision 0.29.2 |

复现命令：

```bash
uv venv --python 3.11 <clean>/.venv
uv pip install --python <clean>/.venv/Scripts/python.exe "D:/Develop/av_align[test,transformers]"
cd <任意仓库外目录>
<clean>/.venv/Scripts/python.exe -m pytest D:/Develop/av_align/tests -q   # 20 passed
```

## 验证中发现并已修复的发布元数据缺陷

| # | 缺陷 | 修复 |
|---|---|---|
| F1 | `transformers` extra 缺 `pillow`/`torchvision`——干净环境后端测试报"Missing optional dependencies: torchvision, Pillow"（主开发环境因手工补装而未暴露） | pyproject `transformers` extra 补 `pillow>=9.0`、`torchvision>=0.15`，重装复跑通过 |
| F2 | pyproject `version = "0.1.0"` 与包内 `__version__ = "0.2.0"`、CITATION.cff 不一致 | pyproject 升至 0.2.0，重装复核 |

两条均为发布元数据问题、非代码缺陷；修复后以 0.2.0 重装复跑 20/20 通过。

## 范围说明

本次验证覆盖 L1 全部测试 + 后端 A 的 packaging/snapshot 级测试（CPU）。Colab GPU 侧验证独立归档于工作区 `artifacts/音画时间机制/p0-01-colab-证据/20261115/`（A1–A6 全过）；MLX 侧（后端 B）未验证（见 scope-limitations.md）。
