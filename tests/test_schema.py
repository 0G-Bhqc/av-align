"""schema av_align.layout/1 契约测试：哈希敏感性、roundtrip、损坏检测。"""

from __future__ import annotations

import copy

import pytest

from av_align import LayoutError, LayoutPlan, build_layout, make_synthetic_event_fixture


def _plan():
    fx = make_synthetic_event_fixture()
    return build_layout(fx.video, fx.audio), fx


def test_schema_id_frozen():
    plan, _ = _plan()
    d = plan.to_dict()
    assert d["schema"] == "av_align.layout/1"
    assert d["hash_algorithm"] == "sha256-canonical-json-v1"
    assert d["time_units"] == "seconds_float64"


def test_hash_sensitive_to_any_field():
    plan, fx = _plan()
    base = plan.to_dict()

    variants = []
    # 视频字段改动
    v = copy.deepcopy(base); v["video"]["n_frames"] += 1
    variants.append(v)
    # 音频字段改动
    a = copy.deepcopy(base); a["audio"]["chunk_ms"] += 1.0
    variants.append(a)
    # 交错改动
    s = copy.deepcopy(base); s["interleaving"]["sequence"] = list(reversed(s["interleaving"]["sequence"]))
    variants.append(s)
    # 配对改动
    p = copy.deepcopy(base); p["pairing"][0]["audio_chunks"] = list(p["pairing"][0]["audio_chunks"]) + [999]
    variants.append(p)
    # 干预记录改动
    iv = copy.deepcopy(base); iv["interventions"] = [{"op": "passthrough", "params": {}}]
    variants.append(iv)

    import hashlib, json
    def h(d):
        payload = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    base_h = h({k: v for k, v in base.items() if k not in ("hash",)})
    assert base_h == plan.hash_
    for v in variants:
        assert h({k: val for k, val in v.items() if k not in ("hash",)}) != plan.hash_


def test_roundtrip_and_tamper_detection():
    plan, _ = _plan()
    d = plan.to_dict()
    restored = LayoutPlan.from_dict(copy.deepcopy(d))
    assert restored.hash_ == plan.hash_

    tampered = copy.deepcopy(d)
    tampered["video"]["frames"][0]["t_start"] += 1e-6
    with pytest.raises(LayoutError):
        LayoutPlan.from_dict(tampered)

    wrong_schema = copy.deepcopy(d)
    wrong_schema["schema"] = "av_align.layout/0"
    with pytest.raises(LayoutError):
        LayoutPlan.from_dict(wrong_schema)


def test_meta_not_in_hash_input_is_recorded():
    plan, _ = _plan()
    assert plan.meta["created_by"].startswith("av_align ")
