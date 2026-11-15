# schema `av_align.layout/1`

冻结：2026-11-15（G1，对齐研发方案 v2 §3.2）。字段改动必须升版本号；旧结果按旧 schema 读，禁止静默回填。

## 顶层字段（canonical dict）

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema` | str | 固定 `"av_align.layout/1"` |
| `hash_algorithm` | str | `"sha256-canonical-json-v1"` |
| `time_units` | str | `"ns_int64"`（共享媒体原点上的整数纳秒，半开区间 [start, end)） |
| `media_units` | object | **源时间**：`{"video": [单元记录…], "audio": [单元记录…]}`，按 (modality, t_start, unit_id) 排序 |
| `request` | object | 见下 |
| `pairing` | list | 逐视频单元配对表（基于**目标**区间） |
| `meta` | object | `created_by` 等 |
| `hash` | str | **内容哈希**：覆盖 media_units + request.target + request.expected_sequence + pairing；**不含** request.interventions 与 meta |

## 媒体单元记录

| 字段 | 说明 |
|---|---|
| `unit_id` | 稳定单元 ID（轨道内唯一） |
| `modality` | `"video"` \| `"audio"` |
| `t_start` / `t_end` | 整数纳秒（源时间） |
| `time_source` | `"container_pts"` \| `"decoder"` \| `"extrapolated"`（v2：缺失时间仅可按显式配置推断） |
| `source_ref` | 内容引用：视频为帧下标或 (首帧, 末帧)；音频为样本区间 (s0, e0) |

轨道粒度由调用方定：原始帧或后端 patch（对齐 Transformers/MLX 的 grid t 维，temporal_patch_size=2）。

## request

| 字段 | 说明 |
|---|---|
| `interventions` | 干预记录（provenance，canonical 顺序）：`{"op": "passthrough"\|"shift_both"\|"reindex"\|"swap", "params": …}`。**不在内容哈希内**——identity 与空干预哈希一致（v2 §4.1 性质 1）；有效变更必然体现在 target/pairing 上 |
| `target` | **目标时间**：`{"video": [单元记录…], "audio": [单元记录…]}`，干预后的区间；无干预时与 media_units 逐字段一致 |
| `expected_sequence` | L1 期望媒体序列 `[{modality, unit_id, t_start, t_end}]`，供后端快照核验（后端实际顺序以快照为准） |
| `sequence_opts` | `video_block_s`（值域分窗长）/ `within_block_order`（"video_first" = 后端实际语义，"audio_first" = 窗内时间严格） |

## 干预语义（v2 §3.3）

| 算子 | 目标区间变更 |
|---|---|
| `passthrough` | 无（记录 provenance） |
| `shift_both(delta_ns)` | 两路同加整数纳秒偏移 |
| `reindex(side, mapping)` | 按稳定单元 ID 改写指定单元目标起点，时长保持；未匹配 ID / 非整数报错；部分覆盖合法 |
| `swap(side, pair)` | 两单元目标区间互换（内容归属交换），源时间不变，两次 swap 可逆 |

## pairing

`[{video_unit, audio_units: [unit_id…], t_start, t_end}]`。重叠判定：闭开区间重叠 > 1 纳秒（端点相接视为无重叠，v2）。配对表在共同平移下逐项不变（性质 2 依据）。

## 哈希用途与边界

- 双后端对照（A/B 产出逐字段比对）、实验归档引用、篡改检测（`LayoutPlan.from_dict` 读入即重算校验）；
- 篡改 `request.interventions` 不触发哈希失败（设计约束：provenance 不绑定内容哈希；有效变更必然反映在 target/pairing）。

## 后端快照（独立 schema，草案）

`av_align.backend_snapshot/0-draft`（`av_align/adapters/transformers_adapter.py::build_backend_snapshot`）：`time_constants`（25 pos/s、2.0 s/窗、patch=2 帧、音频 40 ms/token）、`special_token_ids`、`media_runs`、`unit_token_map`（unit→token 区间 + 位置值 + 截断规则）、`unverified_fields`（权重 SHA、前向入参快照等显式留空）。待 G2 冻结为 `/1`。
