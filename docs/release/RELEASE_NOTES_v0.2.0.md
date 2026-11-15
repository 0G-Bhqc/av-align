# av_align 0.2.0 发布说明（G4 · 按已验证能力范围）

发布日期 2026-11-15 · 许可 Apache-2.0 · 仓库 `av-align`（远端建立待公开确认）

## 本版本包含

**L1 布局核心**（`av_align.layout/1`，纯 NumPy、CPU 可测）：
- 整数纳秒时间（共享媒体原点）、稳定单元 ID + 时间来源标注（container_pts / decoder / extrapolated）
- 源时间（`media_units`）与请求时间（`request.target`）分离；内容哈希不含干预记录（identity 哈希中性）
- 四个干预算子：`passthrough` / `shift_both` / `reindex`（按稳定 ID，未匹配报错）/ `swap`（内容归属交换，可逆）
- 逐单元音画配对表（共同平移不变）；canonical JSON 内容哈希 + 篡改检测

**后端 A 适配器**（Transformers / Qwen2.5-Omni）：
- packaging 与位置编码的只读重述（`av_align.backend_snapshot/0-draft`）：processor 占位串与 L1 运行序列一致；tiny-config `get_rope_index` 逐运行核验一致
- Colab A100 全前向实测（transformers 4.57.6 @ Qwen2.5-Omni-3B `f75b40e3…`）：原生两次前向位级一致（抖动 0）、显式回放位级一致（差 0）、位置干预（只动 `position_ids`）passthrough 重述 0.0、单路视频 +1s 扰动 0.172、位移 [25] 有效且 npz 独立复核

**测试**：20 项（v2 §4.1 六项性质 + 契约/schema + 后端 A；干净环境非 editable 安装验证通过）

## 已知限制（发布即披露，详见 docs/release/scope-limitations.md）

- 后端 B（mlx-vlm / qwen3_omni_moe）：16 GiB 内存超限（官方最低位宽 4bit = 20.33 GiB 实测）——两阶段 Mac 运行包随证据待执行，load 失败即降级决策证据
- 后端 C：候选清单与打分预填已立，冻结待 P0-03/P2-01
- transformers 5.23 已废弃 `use_audio_in_video`（v5.25 移除）；适配器双路径兼容已实现
- 接口语义敏感性（Q1）：`use_audio_in_video` True/False 改变模型答案——交研究线引用

## 安装

```bash
pip install .            # 仅 L1（numpy）
pip install ".[test]"    # + 测试
pip install ".[transformers]"  # + 后端 A（torch/transformers/pillow/torchvision）
pytest tests -q          # 20 passed（后端四项需 HF 网络）
```

## 引用

见 `CITATION.cff`；引用请注明 schema 版本（av_align.layout/1）、验证后端与模型 revision。
