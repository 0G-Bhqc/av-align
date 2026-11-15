"""v2 方案 §4.1 的六项性质测试（G1 验收核心）。

对齐 v2 契约：整数纳秒、稳定单元 ID、源时间/目标时间分离、swap=内容归属交换、
identity 干预哈希中性。测试 3 的期望由测试内独立实现（重叠公式 + 排序），
不调用被测的 compute_pairing/interleave，避免循环论证。
"""

from __future__ import annotations

import numpy as np
import pytest

from av_align import (
    LayoutError,
    MediaUnit,
    Passthrough,
    ReindexSide,
    ShiftBoth,
    SwapUnits,
    UnitTrack,
    build_layout,
    make_synthetic_event_fixture,
    s_to_ns,
)

FPS = 4.0
CHUNK_MS = 40.0
SR = 16000.0
DUR = 10.0


def _fixture(event_t=3.2):
    return make_synthetic_event_fixture(
        duration_s=DUR, fps=FPS, sample_rate=SR, chunk_ms=CHUNK_MS, event_t=event_t
    )


def _runs(seq):
    out = []
    for s in seq:
        if out and out[-1]["modality"] == s["modality"]:
            out[-1]["n"] += 1
        else:
            out.append({"modality": s["modality"], "n": 1})
    return out


# --- 性质 1：原样通过（v2：空干预与显式 identity 产出相同布局内容和哈希） ------

def test_passthrough_identity():
    fx = _fixture()
    base = build_layout(fx.video, fx.audio)
    passed = build_layout(fx.video, fx.audio, interventions=[Passthrough()])
    assert base["hash"] == passed["hash"]
    # 布局内容逐字段一致；request.interventions 是 provenance，identity 只多一条记录
    assert base["media_units"] == passed["media_units"]
    assert base["request"]["target"] == passed["request"]["target"]
    assert base["request"]["expected_sequence"] == passed["request"]["expected_sequence"]
    assert base["pairing"] == passed["pairing"]
    # identity 干预的目标区间与源时间逐字段一致
    assert passed["request"]["target"] == passed["media_units"]
    assert passed["request"]["interventions"] == [{"op": "passthrough", "params": {}}]


# --- 性质 2：共同平移（两路同加偏移，相对量与重叠配对保留） -------------------

def test_common_shift_equivariance():
    fx = _fixture()
    dt = s_to_ns(0.7)  # 呼应媒体探针的 700ms 工况
    base = build_layout(fx.video, fx.audio)
    shifted = build_layout(fx.video, fx.audio, interventions=[ShiftBoth(dt)])

    seq_b = [(s["modality"], s["unit_id"]) for s in base["request"]["expected_sequence"]]
    seq_s = [(s["modality"], s["unit_id"]) for s in shifted["request"]["expected_sequence"]]
    assert seq_b == seq_s

    tb = [s["t_start"] for s in base["request"]["expected_sequence"]]
    ts = [s["t_start"] for s in shifted["request"]["expected_sequence"]]
    for i in range(1, len(tb)):
        assert ts[i] - ts[i - 1] == tb[i] - tb[i - 1]  # 整数纳秒：精确相等，无浮点误差

    for sb, ss in zip(base["request"]["expected_sequence"], shifted["request"]["expected_sequence"]):
        assert ss["t_start"] == sb["t_start"] + dt
        assert ss["t_end"] == sb["t_end"] + dt

    pb = {(p["video_unit"], tuple(p["audio_units"])) for p in base["pairing"]}
    ps = {(p["video_unit"], tuple(p["audio_units"])) for p in shifted["pairing"]}
    assert pb == ps  # 配对表逐项不变


# --- 性质 3：单路重索引（预先手算映射逐字段比对） -----------------------------

def _independent_pairing(video_spans, audio_spans):
    """测试内独立实现：unit_id → 与其重叠的音频 unit_id 集合（重叠 > 1ns）。"""
    out = {}
    for vid, (v0, v1) in video_spans.items():
        acc = []
        for aid, (a0, a1) in sorted(audio_spans.items()):
            if min(v1, a1) - max(v0, a0) > 1:
                acc.append(aid)
        out[vid] = tuple(acc)
    return out


def test_single_side_reindex_predicted():
    fx = _fixture()
    dt = s_to_ns(0.35)
    mapping = {u.unit_id: u.t_start + dt for u in fx.video.units}
    base = build_layout(fx.video, fx.audio)
    re = build_layout(fx.video, fx.audio, interventions=[ReindexSide("video", mapping)])

    # 音频目标区间完全不动
    assert base["request"]["target"]["audio"] == re["request"]["target"]["audio"]
    # 视频目标区间 = 原区间起点 + dt，时长保持；源时间不变
    src = {u["unit_id"]: (u["t_start"], u["t_end"]) for u in re["media_units"]["video"]}
    tgt = {u["unit_id"]: (u["t_start"], u["t_end"]) for u in re["request"]["target"]["video"]}
    for uid, new_start in mapping.items():
        assert tgt[uid][0] == new_start
        assert tgt[uid][1] - tgt[uid][0] == src[uid][1] - src[uid][0]
    assert src == {u["unit_id"]: (u["t_start"], u["t_end"]) for u in base["media_units"]["video"]}

    # 与测试内独立实现比对（配对）
    audio_spans = {u["unit_id"]: (u["t_start"], u["t_end"]) for u in re["request"]["target"]["audio"]}
    got_pair = {p["video_unit"]: tuple(p["audio_units"]) for p in re["pairing"]}
    assert got_pair == _independent_pairing(tgt, audio_spans)

    # 绑定随干预可预测：覆盖 event_t+dt 的音频块 ∈ 闪帧目标区间的配对集
    target_chunk = next(
        uid for uid, (a0, a1) in audio_spans.items() if a0 <= s_to_ns(fx.event_t_s) + dt < a1
    )
    assert target_chunk in got_pair[fx.flash_unit_id]


def test_reindex_invalid_mappings_raise():
    fx = _fixture()
    with pytest.raises(LayoutError):
        build_layout(fx.video, fx.audio, interventions=[ReindexSide("video", {"不存在": 0})])  # 未匹配 ID
    with pytest.raises(LayoutError):
        build_layout(fx.video, fx.audio, interventions=[ReindexSide("video", {fx.video.units[0].unit_id: 0.5})])  # 非整数纳秒

    # v2 语义：部分覆盖合法——只改写指定单元的目标时间，其余单元逐字段不动
    mapped = fx.video.units[3].unit_id
    re = build_layout(fx.video, fx.audio, interventions=[ReindexSide("video", {mapped: fx.video.units[3].t_start + s_to_ns(1.0)})])
    tgt = {u["unit_id"]: (u["t_start"], u["t_end"]) for u in re["request"]["target"]["video"]}
    for u in fx.video.units:
        if u.unit_id == mapped:
            assert tgt[u.unit_id][0] == u.t_start + s_to_ns(1.0)
        else:
            assert tgt[u.unit_id] == (u.t_start, u.t_end)


# --- 性质 4：输入顺序稳定（打乱帧记录，布局与哈希一致） ------------------------

def test_input_order_stability():
    fx = _fixture()
    rng = np.random.default_rng(17)
    order = rng.permutation(len(fx.video.units))
    shuffled = UnitTrack(units=[fx.video.units[i] for i in order], modality="video")
    base = build_layout(fx.video, fx.audio)
    other = build_layout(shuffled, fx.audio)
    assert base["hash"] == other["hash"]
    assert base == other


# --- 性质 5：事件配对和换位（配对 + 内容归属交换 + 逆变换） --------------------

def test_event_pairing_and_swap():
    fx = _fixture(event_t=3.2)
    plan = build_layout(fx.video, fx.audio)
    pairing = {p["video_unit"]: tuple(p["audio_units"]) for p in plan["pairing"]}
    assert fx.beep_unit_id in pairing[fx.flash_unit_id]

    # swap：两个音频单元目标区间互换（内容归属交换），源时间不变
    other = fx.audio.units[0].unit_id if fx.audio.units[0].unit_id != fx.beep_unit_id else fx.audio.units[5].unit_id
    swapped = build_layout(fx.video, fx.audio, interventions=[SwapUnits("audio", fx.beep_unit_id, other)])
    src = {u["unit_id"]: (u["t_start"], u["t_end"]) for u in swapped["media_units"]["audio"]}
    tgt = {u["unit_id"]: (u["t_start"], u["t_end"]) for u in swapped["request"]["target"]["audio"]}
    a, b = fx.beep_unit_id, other
    assert tgt[a] == src[b] and tgt[b] == src[a]
    assert tgt[a][1] - tgt[a][0] == src[a][1] - src[a][0]  # 时长保持
    # 配对随目标区间改变：提示音不再与闪帧配对
    pairing2 = {p["video_unit"]: tuple(p["audio_units"]) for p in swapped["pairing"]}
    assert fx.beep_unit_id not in pairing2[fx.flash_unit_id]

    # 逆变换：两次 swap 恢复原目标区间 → 哈希与无干预一致（干预记录不在哈希内）
    restored = build_layout(
        fx.video,
        fx.audio,
        interventions=[SwapUnits("audio", fx.beep_unit_id, other), SwapUnits("audio", fx.beep_unit_id, other)],
    )
    assert restored["hash"] == plan["hash"]


# --- 性质 6：重采样稳定（冻结采样配置，同一事件对应关系保持） ------------------

def test_resampling_stability():
    event_t = 3.2
    for fps, chunk_ms in [(4.0, 40.0), (8.0, 20.0), (2.0, 80.0)]:
        fx = make_synthetic_event_fixture(
            duration_s=DUR, fps=fps, sample_rate=SR, chunk_ms=chunk_ms, event_t=event_t
        )
        plan = build_layout(fx.video, fx.audio)
        pairing = {p["video_unit"]: tuple(p["audio_units"]) for p in plan["pairing"]}
        assert fx.beep_unit_id in pairing[fx.flash_unit_id], (fps, chunk_ms)


# --- 契约（v2 §3.2） ----------------------------------------------------------

def test_no_input_mutation():
    fx = _fixture()
    src_before = [(u.unit_id, u.t_start, u.t_end, u.source_ref) for u in fx.video.units] + [
        (u.unit_id, u.t_start, u.t_end, u.source_ref) for u in fx.audio.units
    ]
    build_layout(
        fx.video,
        fx.audio,
        interventions=[
            ShiftBoth(s_to_ns(1.0)),
            SwapUnits("video", fx.video.units[0].unit_id, fx.video.units[5].unit_id),
        ],
    )
    src_after = [(u.unit_id, u.t_start, u.t_end, u.source_ref) for u in fx.video.units] + [
        (u.unit_id, u.t_start, u.t_end, u.source_ref) for u in fx.audio.units
    ]
    assert src_before == src_after


def test_invalid_inputs_raise():
    fx = _fixture()
    with pytest.raises(LayoutError):
        UnitTrack(units=[fx.video.units[0], fx.video.units[0]], modality="video")  # ID 重复
    with pytest.raises(LayoutError):
        MediaUnit("x", "video", 0, 100, time_source="拍脑袋")  # 未知时间来源
    with pytest.raises(LayoutError):
        MediaUnit("x", "video", 100, 100)  # 时长非正
    with pytest.raises(LayoutError):
        MediaUnit("x", "video", 0.5, 100)  # 非整数纳秒
    with pytest.raises(LayoutError):
        UnitTrack.from_unit_times([(1.0, 0.5)], "audio")  # 区间无效
    with pytest.raises(LayoutError):
        UnitTrack.from_unit_times([(0.0, 0.1), (0.05, 0.15)], "audio")  # 音频单元重叠
    with pytest.raises(LayoutError):
        build_layout(fx.video, fx.audio, interventions=[SwapUnits("video", fx.video.units[0].unit_id, "不存在")])


def test_sequential_and_block_modes():
    fx = _fixture()
    plan_seq = build_layout(fx.video, fx.audio, use_audio_in_video=False)
    mods = [s["modality"] for s in plan_seq["request"]["expected_sequence"]]
    assert mods == ["video"] * len(fx.video.units) + ["audio"] * len(fx.audio.units)

    # 分块模式（后端重述）：patch 槽位 + 40ms 音频块，2 秒窗、块内视频先行
    video = UnitTrack.from_patch_grid(n_patches=6, fps=FPS)  # 12 原始帧 @4fps = 3s
    audio = UnitTrack.from_unit_times([(i * 0.04, (i + 1) * 0.04) for i in range(75)], "audio")
    plan_blk = build_layout(video, audio, video_block_s=2.0, within_block_order="video_first")
    assert _runs(plan_blk["request"]["expected_sequence"]) == [
        {"modality": "video", "n": 4},
        {"modality": "audio", "n": 50},
        {"modality": "video", "n": 2},
        {"modality": "audio", "n": 25},
    ]
