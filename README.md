# av_align · 音画时间对齐算子

多模态大模型的音画时间对齐算子（L1 布局核心 + 干预算子 + 后端适配器）。立项依据：[多模态音画时间对齐算子研发与发布方案](https://github.com/)（工作区 `多模态音画时间对齐算子研发方案.md`，2026-11-15 **v2**）。

## 定位

把**源媒体时间、请求施加的时间布局、模型实际收到的位置和顺序**记录为可核对的显式布局（schema `av_align.layout/1`）：整数纳秒、稳定单元 ID + 时间来源、源时间/目标时间分离、四个干预算子、音画配对表。布局计算是平台无关的纯数据变换（NumPy）；模型后端只消费布局，不改权重。

```
L1 布局核心（本包本体，纯 NumPy，CPU 可测）
  ↓ av_align.layout/1（含内容哈希）+ 后端快照（av_align.backend_snapshot/0-draft）
L2 后端适配器（transformers / mlx；只转发，不携带自己的对齐逻辑）
  ↓
L3 消费方（音视频推理诊断研究线：错例回放、干预接口、时间重索引修复）
```

## 快速上手

```python
from av_align import build_layout, UnitTrack, ReindexSide, ShiftBoth, SwapUnits, Passthrough

video = UnitTrack.from_patch_grid(n_patches=20, fps=4.0)   # 10 s @4fps，patch 粒度（2 帧/patch）
audio = UnitTrack.from_unit_times([(i * 0.04, (i + 1) * 0.04) for i in range(250)], "audio")

plan = build_layout(video, audio, video_block_s=2.0, within_block_order="video_first")
plan["pairing"]   # 逐单元配对表（基于目标区间）：绑定检验与干预定位的索引
plan["hash"]      # 内容哈希（不含干预记录）：双后端对照与实验归档

# 干预：只移一路的目标时间（主干预）；两路共同平移（等变对照）；内容归属交换（换位）
plan2 = build_layout(video, audio, interventions=[
    ReindexSide("video", {u.unit_id: u.t_start + 350_000_000 for u in video.units}),  # +0.35 s（ns）
])
```

## 性质测试（v2 §4.1 G1 验收核心，`tests/`，20 项全过）

1. 原样通过：空干预与显式 identity 产出相同布局内容和哈希
2. 共同平移：两路同加偏移，全部相对量与重叠配对保留（整数纳秒精确相等）
3. 单路重索引：预先手算映射逐字段比对；未匹配 ID / 非整数明确报错；部分覆盖合法
4. 输入顺序稳定：打乱帧记录输入顺序，布局与哈希一致
5. 事件配对和换位：合成事件（已知时刻闪帧 × 提示音）配对；swap 内容归属交换可逆
6. 重采样稳定：fps / chunk_ms 粒度变化下事件对应关系保持

后端 A（Transformers 5.23 / Qwen2.5-Omni）双级只读重述已通过（CPU、无权重）：processor 占位串（后端自身代码）与 L1 期望序列一致；tiny-config `get_rope_index` 时间位置逐运行核验一致（视频 patch 值截断、音频稠密）。

## 复现

```bash
# Windows / Git Bash；macOS 同理（建议 venv 放本机非同步路径）
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[test]"       # L1 + 性质测试，纯 CPU
.venv/Scripts/python.exe -m pytest tests -q                          # 20 项（性质 6 + 契约/schema 6 + 后端 A 4 + 常量 1 + 分块 1 等）

# 后端 A 四项需 HF 网络（processor 配置文件，数 MB，不下载权重）；离线跳过：
AV_ALIGN_OFFLINE=1 .venv/Scripts/python.exe -m pytest tests -q
```

## 仓库纪律

- 仓库在本地磁盘（`D:\Develop\av_align`），不进 Google Drive 同步目录；Drive 只放文档与归档。
- schema 字段改动必须升版本号；旧结果按旧 schema 读，禁止静默回填。
- Apache-2.0；解码依赖逐项审许可后才能进发行目录（G4 门）。

## 状态（2026-11-15）

- [x] P1-01 L1 对齐 v2 契约 + 六项性质测试（docs/schema-layout-1.md）
- [x] P0-01 Transformers 5.23.0 实测勘察（v2 §1 A 轨补充；4.57.6 仍是 G2 全前向版本依据）——记录在工作区 `artifacts/音画时间机制/P0-01-transformers-勘察记录.md`
- [x] P0-02 mlx-vlm 0.7.12 勘察：确认无 qwen2_5_omni（v2 §1 B 轨）；B 适配器改 qwen3_omni_moe
- [x] P1-02 后端 A 双级只读重述 + v2 后端快照记录（`av_align.backend_snapshot/0-draft`）
- [ ] P0-03 原生/显式回放等价与错例回放（需 3B 权重，Colab CUDA 或本机下载）
- [ ] P2-01 后端 B Metal 执行（`av_align/adapters/mlx_adapter.py::verify_on_metal` 内置步骤）
- [ ] P2-02 后端 C 选型冻结（docs/backend-c-candidates.md）
- [ ] G4 发布（CITATION.cff、安装验证、复现命令）
