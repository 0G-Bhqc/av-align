# av_align · 音画时间对齐算子

多模态大模型的音画时间对齐算子（L1 布局核心 + 干预算子 + 后端适配器）。立项依据：[多模态音画时间对齐算子研发与发布方案](https://github.com/)（工作区 `多模态音画时间对齐算子研发方案.md`，2026-11-15 v1）。

## 定位

输入视频帧时间戳与音频块时间戳，输出**显式时间对齐布局**（schema `av_align.layout/1`）：逐 token 模态、时间区间、交错顺序、音画配对表，外加四个干预算子。布局计算是平台无关的纯数据变换（NumPy）；模型后端只消费布局，不改权重。

```
L1 布局核心（本包本体，纯 NumPy，CPU 可测）
  ↓ 布局计划（av_align.layout/1，含哈希）
L2 后端适配器（transformers / mlx；只转发，不携带自己的对齐逻辑）
  ↓
L3 消费方（音视频推理诊断研究线：干预接口、时间重索引修复）
```

## 快速上手

```python
from av_align import build_layout, VideoTrack, AudioTrack, ReindexSide, ShiftBoth

video = VideoTrack.from_fps(n_frames=40, fps=4.0)          # 10 s @ 4 fps
audio = AudioTrack.from_chunk_times([(i * 0.04, (i + 1) * 0.04) for i in range(250)])

plan = build_layout(video, audio, use_audio_in_video=True)
plan.pairing      # 逐帧配对表：绑定检验与干预定位的索引
plan.hash_        # sha256-canonical-json-v1，供双后端对照与实验归档

# 干预：只移一路的时间位置（主干预），两路共同平移（等变对照）
plan2 = build_layout(video, audio, interventions=[
    ReindexSide("video", {f.index: f.t_start + 0.35 for f in video.frames}),
])
```

## 性质测试（G1 验收核心，`tests/`）

1. 原样通过等价（L1 部分：identity 干预只增 provenance 记录，内容逐字段不变；后端等价在 P1-02 适配器验证）
2. 共同平移等变：相对量逐项不变，绝对时间恰移 Δ
3. 单路平移可预测：期望由测试内独立实现比对（非循环论证）
4. 帧序无关：带时间戳乱序帧输入产出同一布局（哈希一致）
5. 合成事件绑定夹具：已知时刻闪帧 × 提示音落入相同时间槽
6. 重采样稳定：fps / chunk_ms 粒度变化下事件槽包含关系保持

## 仓库纪律

- 仓库在本地磁盘（`D:\Develop\av_align`），不进 Google Drive 同步目录；Drive 只放文档与归档。
- schema 字段改动必须升版本号；旧结果按旧 schema 读，禁止静默回填。
- Apache-2.0；解码依赖逐项审许可后才能进发行目录（G4 门）。

## 状态

- [x] P1-01 L1 布局核心 + 六项性质测试（2026-11-15，14 tests passed）
- [x] P0-02 mlx-vlm 0.7.12 勘察（记录见工作区 `artifacts/音画时间机制/P0-02-mlxvlm-勘察记录.md`）
- [ ] P0-01 Transformers 勘察 → P1-02 后端 A 适配器 + CPU 级原样通过验证
- [ ] P2-01 后端 B 适配器（qwen3_omni_moe 路线调整后待 Apple Silicon 执行）
- [ ] P2-02 后端 C 选型冻结
- [ ] G4 发布（CITATION.cff、安装验证、复现命令）
