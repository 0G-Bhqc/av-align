"""方案 4.4 的六项性质测试（G1 验收核心）。

每项测试对应方案条目，失败即 G1 不通过。测试 3 的期望由测试内独立实现
（直接重叠公式 + 排序），不调用被测的 compute_pairing/interleave，避免循环论证。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from av_align import (
    AudioTrack,
    LayoutError,
    Passthrough,
    ReindexSide,
    ShiftBoth,
    SwapTimeSlots,
    VideoTrack,
    build_layout,
    make_synthetic_event_fixture,
)

FPS = 4.0
CHUNK_MS = 40.0
SR = 16000.0
DUR = 10.0


def _fixture(event_t=3.2):
    return make_synthetic_event_fixture(
        duration_s=DUR, fps=FPS, sample_rate=SR, chunk_ms=CHUNK_MS, event_t=event_t
    )


# --- 性质 1：原样通过等价（L1 部分） -----------------------------------------

def test_passthrough_identity():
    fx = _fixture()
    base = build_layout(fx.video, fx.audio)
    passed = build_layout(fx.video, fx.audio, interventions=[Passthrough()])
    # 布局内容逐字段一致；干预记录是 provenance，原样通过只在 interventions 里多一条 identity 记录
    for key in ("video", "audio", "interleaving", "pairing"):
        assert base.to_dict()[key] == passed.to_dict()[key]
    assert passed.interventions == [{"op": "passthrough", "params": {}}]


# --- 性质 2：共同平移等变 -----------------------------------------------------

def test_common_shift_equivariance():
    fx = _fixture()
    dt = 0.7
    base = build_layout(fx.video, fx.audio)
    shifted = build_layout(fx.video, fx.audio, interventions=[ShiftBoth(dt)])

    # 相对顺序逐项不变
    seq_b = [(s["modality"], s["ref"]) for s in base.interleaving["sequence"]]
    seq_s = [(s["modality"], s["ref"]) for s in shifted.interleaving["sequence"]]
    assert seq_b == seq_s

    # 相邻 token 时间差逐项不变（等预算的相对时间关系）
    tb = [s["t_start"] for s in base.interleaving["sequence"]]
    ts = [s["t_start"] for s in shifted.interleaving["sequence"]]
    assert len(tb) == len(ts)
    for i in range(1, len(tb)):
        assert ts[i] - ts[i - 1] == pytest.approx(tb[i] - tb[i - 1], abs=1e-12)

    # 绝对时间恰好平移 dt
    for sb, ss in zip(base.interleaving["sequence"], shifted.interleaving["sequence"]):
        assert ss["t_start"] == pytest.approx(sb["t_start"] + dt, abs=1e-12)
        assert ss["t_end"] == pytest.approx(sb["t_end"] + dt, abs=1e-12)

    # 配对表逐项不变（模态与引用集合）
    pb = {(p["video_frame"], tuple(p["audio_chunks"])) for p in base.pairing}
    ps = {(p["video_frame"], tuple(p["audio_chunks"])) for p in shifted.pairing}
    assert pb == ps


# --- 性质 3：单路平移可预测 ---------------------------------------------------

def _independent_layout(frames, chunks):
    """测试内独立实现：直接重叠公式 + 排序，不调用 av_align 的对应函数。"""
    pairing = {}
    for f in sorted(frames, key=lambda x: (x.t_start, x.index)):
        acc = []
        for c in sorted(chunks, key=lambda x: x.index):
            if min(f.t_end, c.t_end) - max(f.t_start, c.t_start) > 1e-9:
                acc.append(c.index)
        pairing[f.index] = tuple(acc)
    events = [(f.t_start, 0, "video", f.index) for f in frames] + [
        (c.t_start, 1, "audio", c.index) for c in chunks
    ]
    events.sort(key=lambda e: (e[0], e[1], e[3]))
    return pairing, [(m, r) for _, _, m, r in events]


def test_single_side_shift_predictable():
    fx = _fixture()
    dt = 0.35
    base = build_layout(fx.video, fx.audio)
    re = build_layout(
        fx.video, fx.audio, interventions=[ReindexSide("video", {f.index: f.t_start + dt for f in fx.video.frames})]
    )

    # 音频侧完全不动
    assert base.audio["chunks"] == re.audio["chunks"]

    # 视频帧时间槽 = 原槽 + dt（时长保持）
    for fb, fr in zip(base.video["frames"], re.video["frames"]):
        assert fr["index"] == fb["index"]
        assert fr["t_start"] == pytest.approx(fb["t_start"] + dt, abs=1e-12)
        assert fr["t_end"] - fr["t_start"] == pytest.approx(fb["t_end"] - fb["t_start"], abs=1e-12)

    # 与测试内独立实现比对
    frames = [
        type("F", (), {"index": fr["index"], "t_start": fr["t_start"], "t_end": fr["t_end"]})()
        for fr in re.video["frames"]
    ]
    chunks = [
        type("C", (), {"index": c["index"], "t_start": c["t_start"], "t_end": c["t_end"]})()
        for c in re.audio["chunks"]
    ]
    exp_pair, exp_seq = _independent_layout(frames, chunks)
    got_pair = {p["video_frame"]: tuple(p["audio_chunks"]) for p in re.pairing}
    assert got_pair == exp_pair
    got_seq = [(s["modality"], s["ref"]) for s in re.interleaving["sequence"]]
    assert got_seq == exp_seq

    # 绑定随干预可预测地改变：闪帧对应的音频块集合 = 覆盖 event_t+dt 的块
    fx2_chunks = [c for c in re.audio["chunks"]]
    target = next(c["index"] for c in fx2_chunks if c["t_start"] <= fx.event_t + dt < c["t_end"])
    frame_slot = next(fr for fr in re.video["frames"] if fr["index"] == fx.flash_frame_index)
    got = {p["video_frame"]: tuple(p["audio_chunks"]) for p in re.pairing}[fx.flash_frame_index]
    assert target in got


# --- 性质 4：帧序无关 ---------------------------------------------------------

def test_frame_order_invariance():
    fx = _fixture()
    rng = np.random.default_rng(17)
    order = rng.permutation(len(fx.video.frames))
    shuffled = VideoTrack(frames=[fx.video.frames[i] for i in order], fps=fx.video.fps)
    base = build_layout(fx.video, fx.audio)
    other = build_layout(shuffled, fx.audio)
    assert base.hash_ == other.hash_
    assert base.to_dict() == other.to_dict()


# --- 性质 5：事件绑定夹具 -----------------------------------------------------

def test_synthetic_event_binding():
    fx = _fixture(event_t=3.2)
    plan = build_layout(fx.video, fx.audio)
    pairing = {p["video_frame"]: tuple(p["audio_chunks"]) for p in plan.pairing}
    # 闪帧与提示音块落入相同（重叠）时间槽：提示音块在闪帧的配对集合中
    assert fx.beep_chunk_index in pairing[fx.flash_frame_index]

    # 两事件情形：各自绑定各自的时间槽
    from av_align import add_beep_at

    fx2 = make_synthetic_event_fixture(duration_s=DUR, fps=FPS, sample_rate=SR, chunk_ms=CHUNK_MS, event_t=2.5)
    late_frame, late_chunk = add_beep_at(fx2, t=7.5)
    plan2 = build_layout(fx2.video, fx2.audio)
    p2 = {p["video_frame"]: tuple(p["audio_chunks"]) for p in plan2.pairing}
    assert late_chunk in p2[late_frame]
    # 两事件不同帧不同块，互不绑定
    assert fx2.flash_frame_index != late_frame
    assert fx2.beep_chunk_index not in p2[late_frame]
    assert late_chunk not in p2[fx2.flash_frame_index]


# --- 性质 6：重采样稳定 -------------------------------------------------------

def test_resampling_stability():
    event_t = 3.2
    cfgs = [
        (4.0, 40.0),
        (8.0, 20.0),   # 更细的时间粒度
        (2.0, 80.0),   # 更粗的时间粒度
    ]
    for fps, chunk_ms in cfgs:
        fx = make_synthetic_event_fixture(
            duration_s=DUR, fps=fps, sample_rate=SR, chunk_ms=chunk_ms, event_t=event_t
        )
        plan = build_layout(fx.video, fx.audio)
        pairing = {p["video_frame"]: tuple(p["audio_chunks"]) for p in plan.pairing}
        # 事件槽包含关系在两种粒度下均成立：覆盖 event_t 的块 ∈ 覆盖 event_t 的帧的配对集
        assert fx.beep_chunk_index in pairing[fx.flash_frame_index], (fps, chunk_ms)


# --- 契约测试（非六项，随附） -------------------------------------------------

def test_no_input_mutation():
    fx = _fixture()
    frames_before = [(f.index, f.t_start, f.t_end) for f in fx.video.frames]
    chunks_before = [(c.index, c.t_start, c.t_end, c.sample_span) for c in fx.audio.chunks]
    build_layout(fx.video, fx.audio, interventions=[ShiftBoth(1.0), SwapTimeSlots("video", 0, 5)])
    assert frames_before == [(f.index, f.t_start, f.t_end) for f in fx.video.frames]
    assert chunks_before == [(c.index, c.t_start, c.t_end, c.sample_span) for c in fx.audio.chunks]


def test_invalid_inputs_raise():
    fx = _fixture()
    with pytest.raises(LayoutError):
        build_layout(VideoTrack.from_fps(2, FPS, t0=float("nan")), fx.audio)
    with pytest.raises(LayoutError):
        AudioTrack.from_chunk_times([(1.0, 0.5)])  # 时长非正
    with pytest.raises(LayoutError):
        # 重索引未全覆盖
        build_layout(fx.video, fx.audio, interventions=[ReindexSide("video", {0: 0.0})])
    with pytest.raises(LayoutError):
        build_layout(fx.video, fx.audio, interventions=[SwapTimeSlots("video", 0, 999)])
    with pytest.raises(LayoutError):
        # 音频块乱序
        AudioTrack.from_chunk_times([(0.0, 0.04), (0.0, 0.04)])


def test_empty_and_sequential_modes():
    empty_v = VideoTrack(frames=[], fps=FPS)
    fx = _fixture()
    plan = build_layout(empty_v, fx.audio)
    assert plan.video["n_frames"] == 0
    assert all(s["modality"] == "audio" for s in plan.interleaving["sequence"])
    plan2 = build_layout(fx.video, fx.audio, use_audio_in_video=False)
    assert plan2.interleaving["mode"] == "sequential"
    seq = plan2.interleaving["sequence"]
    assert [s["modality"] for s in seq] == ["video"] * len(fx.video.frames) + ["audio"] * len(fx.audio.chunks)


def test_video_block_interleaving_matches_backend_style():
    fx = _fixture()
    plan = build_layout(fx.video, fx.audio, video_block_s=2.0)
    seq = plan.interleaving["sequence"]
    assert plan.interleaving["video_block_s"] == 2.0
    # 块内视频帧连续，块间按时间插入音频：序列中视频引用按 index 升序出现在各自块内
    video_refs = [s["ref"] for s in seq if s["modality"] == "video"]
    assert video_refs == sorted(video_refs)
