"""schema av_align.layout/1 契约测试：哈希敏感性、roundtrip、篡改检测。

v2 语义：内容哈希覆盖 media_units + target + expected_sequence + pairing；
干预记录为 provenance，**不在哈希内**（identity 哈希中性），篡改干预记录不会触发
哈希校验失败——有效变更必然体现在 target/pairing 上，这是设计约束而非遗漏。
"""

from __future__ import annotations

import copy

import pytest

from av_align import LayoutError, LayoutPlan, Passthrough, ReindexSide, build_layout, make_synthetic_event_fixture, s_to_ns


def _plan():
    fx = make_synthetic_event_fixture()
    return build_layout(fx.video, fx.audio), fx


def test_schema_id_frozen():
    plan, _ = _plan()
    assert plan["schema"] == "av_align.layout/1"
    assert plan["hash_algorithm"] == "sha256-canonical-json-v1"
    assert plan["time_units"] == "ns_int64"


def _content_hash(d):
    import hashlib
    import json

    payload = {
        "schema": d["schema"],
        "hash_algorithm": d["hash_algorithm"],
        "time_units": d["time_units"],
        "media_units": d["media_units"],
        "request": {"target": d["request"]["target"], "expected_sequence": d["request"]["expected_sequence"]},
        "pairing": d["pairing"],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_hash_matches_content_only():
    plan, _ = _plan()
    assert _content_hash(plan) == plan["hash"]

    # 干预记录不在哈希内：加一条 passthrough 不改变哈希（v2 §4.1 性质 1）
    fx, _ = _plan_ret()
    passed = build_layout(fx.video, fx.audio, interventions=[Passthrough()])
    assert passed["hash"] == plan["hash"]

    # 有效干预改变内容 → 哈希改变
    fx2, _ = _plan_ret()
    shifted = build_layout(fx2.video, fx2.audio, interventions=[ReindexSide("video", {u.unit_id: u.t_start + s_to_ns(0.1) for u in fx2.video.units})])
    assert shifted["hash"] != plan["hash"]


def _plan_ret():
    fx = make_synthetic_event_fixture()
    return fx, build_layout(fx.video, fx.audio)


def test_roundtrip_and_tamper_detection():
    _, plan = _plan_ret()
    d = copy.deepcopy(plan)
    restored = LayoutPlan.from_dict(d)
    assert restored.hash_ == plan["hash"]

    tampered = copy.deepcopy(plan)
    tampered["media_units"]["video"][0]["t_start"] += 1_000_000
    with pytest.raises(LayoutError):
        LayoutPlan.from_dict(tampered)

    tampered2 = copy.deepcopy(plan)
    tampered2["pairing"][0]["audio_units"] = list(tampered2["pairing"][0]["audio_units"]) + ["a00000"]
    with pytest.raises(LayoutError):
        LayoutPlan.from_dict(tampered2)

    wrong_schema = copy.deepcopy(plan)
    wrong_schema["schema"] = "av_align.layout/0"
    with pytest.raises(LayoutError):
        LayoutPlan.from_dict(wrong_schema)


def test_hash_sensitive_to_each_content_block():
    _, plan = _plan_ret()
    base_h = plan["hash"]

    variants = []
    v = copy.deepcopy(plan)
    v["media_units"]["video"][0]["t_start"] += 1
    variants.append(("media_units", v))
    v = copy.deepcopy(plan)
    v["request"]["target"]["audio"][0]["t_end"] += 1
    variants.append(("target", v))
    v = copy.deepcopy(plan)
    v["request"]["expected_sequence"] = list(reversed(v["request"]["expected_sequence"]))
    variants.append(("sequence", v))
    v = copy.deepcopy(plan)
    v["pairing"][0]["t_start"] += 1
    variants.append(("pairing", v))

    for name, variant in variants:
        assert variant["hash"] == base_h  # canonical() 未重算时 hash 字段保持原值
        assert _content_hash(variant) != base_h, name  # 内容变化在重算哈希中体现
        with pytest.raises(LayoutError):
            LayoutPlan.from_dict(variant)  # 读入即校验：内容与存储哈希不符则拒绝


def test_time_provenance_recorded():
    _, plan = _plan_ret()
    for u in plan["media_units"]["video"]:
        assert u["time_source"] == "container_pts"
    for u in plan["media_units"]["audio"]:
        assert u["time_source"] == "decoder"
