"""P2-01 · B 侧打包对照（processor-only，无需模型权重）——可在任意 Apple Silicon 上运行。

在 macOS arm64 + MLX 真实环境（GitHub Actions macos-14 runner 或实体 Mac）上：
1. 平台/版本记录（mlx、transformers、macOS arch）；
2. vendored L1 复生成 + SHA 断言（跨平台复现 make_vendor）；
3. 同一合成夹具（与 P0-01 同源，种子 0）→ A/V 容器（系统 ffmpeg）→
   mlx-vlm prepare_omni_inputs(use_audio_in_video=True)；
4. 媒体 token 运行序列 vs L1 期望（pip/spc 从模型 config 实测读取，默认 13/2.0）；
5. 证据落 p2-01-evidence/<日期>/（report.json + input_ids_b.npz + manifest.json）。

软退出：打包不一致登记为差异（证据语义），仅硬错误（导入/处理器失败）才非零退出。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import wave

import numpy as np

L1_SHA256 = "5e9882849343e958630f8ad3683453776d7c9d58bfd2f3cbb1be2b76029b9895"
DEFAULT_MODEL = "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"
FPS, DUR, SR, H, W, EVENT_T = 4.0, 6.0, 16000, 56, 56, 3.2
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = REPO / "p2-01-evidence" / datetime.datetime.now().strftime("%Y%m%d")


def platform_report() -> dict:
    import mlx.core as mx

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "mlx_device": str(mx.default_device()),
    }


def regenerate_vendored() -> dict:
    """运行 scripts/make_vendor.py 并断言 SHA（跨平台复现 make_vendor 的证据）。"""
    r = subprocess.run([sys.executable, str(HERE / "make_vendor.py")],
                       capture_output=True, text=True)
    lines = (r.stdout + r.stderr).strip().splitlines()
    sha_line = next((l for l in lines if l.startswith("sha256")), "")
    got = sha_line.split("->")[-1].strip() if "->" in sha_line else ""
    out = {"returncode": r.returncode, "sha_recorded": got,
           "sha_match": got == L1_SHA256, "raw_tail": lines[-3:]}
    if not out["sha_match"]:
        raise AssertionError(f"vendored L1 SHA 不符（macOS 复现失败）: {out}")
    return out


def fixture_and_container() -> dict:
    import av_align

    rng = np.random.default_rng(0)
    T = int(round(DUR * FPS))
    frames = rng.integers(0, 256, size=(T, H, W, 3), dtype=np.uint8)
    fi = int(round(EVENT_T * FPS))
    frames[fi] = np.uint8(255)
    frames[fi, :, :, 1:] = np.uint8(0)
    samples = rng.standard_normal(int(round(DUR * SR))) * 0.01
    b0, bn = int(round(EVENT_T * SR)), int(round(0.03 * SR))
    samples[b0:b0 + bn] += 0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(bn) / SR)

    video_units = av_align.UnitTrack.from_frame_times(
        [i / FPS for i in range(T)], 1.0 / FPS, time_source="container_pts")
    audio_units = av_align.UnitTrack.from_samples(samples, SR, 40.0, time_source="decoder")
    layout = av_align.build_layout(video_units, audio_units,
                                   video_block_s=2.0, within_block_order="video_first")
    ev_ns = av_align.s_to_ns(EVENT_T)
    flash_uid = next(u.unit_id for u in video_units.units if u.t_start <= ev_ns < u.t_end)
    beep_uid = next(u.unit_id for u in audio_units.units if u.t_start <= ev_ns < u.t_end)
    assert beep_uid in next(p for p in layout["pairing"] if p["video_unit"] == flash_uid)["audio_units"]

    work = pathlib.Path("/tmp") / "av_align_p2_01"
    work.mkdir(exist_ok=True)
    av_path = work / "av.mp4"
    if not av_path.exists():
        import imageio.v2 as imageio
        imageio.mimsave(work / "v.mp4", [f for f in frames], fps=int(FPS))
        with wave.open(str(work / "a.wav"), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
            wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()  # runner 可能无系统 ffmpeg（2026-11-15 实测）
        except Exception:
            ffmpeg_exe = "ffmpeg"
        mux = subprocess.run([ffmpeg_exe, "-y", "-loglevel", "error", "-i", str(work / "v.mp4"),
                              "-i", str(work / "a.wav"), "-c:v", "copy", "-c:a", "aac",
                              "-shortest", str(av_path)], capture_output=True, text=True)
        assert mux.returncode == 0, mux.stderr[-300:]
    return {"layout": layout, "flash_uid": flash_uid, "beep_uid": beep_uid,
            "container": str(av_path), "frames": frames, "samples": samples}


def expected_media_runs(grid_t: int, n_audio_tokens: int, fps: float, tokens_per_frame: int,
                        pip: float, spc: float, tps: float = 2.0):
    window = int(pip * spc)
    step = tps / float(fps) * pip
    vvals = [int(p * step) for p in range(grid_t)]
    nw = max((vvals[-1] // window + 1) if vvals else 0,
             (n_audio_tokens + window - 1) // window if n_audio_tokens else 0)
    runs = []
    for j in range(nw):
        nv = sum(1 for v in vvals if j * window <= v < (j + 1) * window)
        if nv:
            runs.append({"modality": "video", "n_tokens": nv * tokens_per_frame})
        na = min(n_audio_tokens, (j + 1) * window) - min(n_audio_tokens, j * window)
        if na:
            runs.append({"modality": "audio", "n_tokens": na})
    return runs


def runs_of(seq, v_id, a_id):
    out, cur, n = [], None, 0
    for t in seq:
        m = "video" if t == v_id else ("audio" if t == a_id else None)
        if m is None:
            if cur:
                out.append({"modality": cur, "n_tokens": n})
            cur, n = None, 0
        elif cur == m:
            n += 1
        else:
            if cur:
                out.append({"modality": cur, "n_tokens": n})
            cur, n = m, 1
    if cur:
        out.append({"modality": cur, "n_tokens": n})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = ap.parse_args()

    hard_error = None
    evidence: dict = {
        "schema": "av_align.p2_01.processor_check/0-draft",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "channel": "GitHub Actions macos-14（Apple Silicon，真 MLX/Metal 环境；processor-only，无模型权重）",
        "model_repo": args.model,
        "l1_vendored_sha256": L1_SHA256,
    }
    try:
        evidence["platform"] = platform_report()
        evidence["vendor_regeneration"] = regenerate_vendored()

        import transformers  # noqa: F401
        import mlx_vlm

        evidence["versions"] = {
            "transformers": transformers.__version__,
            "mlx_vlm": mlx_vlm.__version__,
        }

        keep = fixture_and_container()
        evidence["layout_hash"] = keep["layout"]["hash"]
        evidence["flash_unit"] = keep["flash_uid"]
        evidence["beep_unit"] = keep["beep_uid"]

        from mlx_vlm.utils import load_processor  # 0.7.12 的 load_processor 在 utils，不在包顶层
        from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs
        from huggingface_hub import snapshot_download
        local_repo = pathlib.Path(snapshot_download(
            args.model, allow_patterns=["*.json", "*.txt", "*.model", "*.jinja",
                                        "tokenizer*", "preprocessor*", "processor*"]))
        # 勘察怪癖（0.7.12）：tokenizer_utils L475 对 model_path 做 Path / 拼接，必须传本地 Path；
        # snapshot 排除 safetensors，只拉配置/tokenizer（约 12 MB）
        processor = load_processor(local_repo)
        cfg_path = pathlib.Path(processor.tokenizer.name_or_path or "") / "config.json"
        pip, spc = 13.0, 2.0
        try:
            import urllib.request  # noqa: F401
            from huggingface_hub import hf_hub_download
            cfg_file = hf_hub_download(args.model, "config.json")
            cfg = json.loads(pathlib.Path(cfg_file).read_text(encoding="utf-8"))
            pip = float(cfg.get("position_id_per_seconds", pip))
            spc = float(cfg.get("seconds_per_chunk", spc))
        except Exception as e:
            evidence["config_read_error"] = repr(e)[:200]
        evidence["time_constants"] = {"position_id_per_seconds": pip, "seconds_per_chunk": spc}

        conversation = [{"role": "user", "content": [
            {"type": "video", "video": keep["container"]},
            {"type": "audio", "audio": str(pathlib.Path(keep["container"]).parent / "a.wav")},
            {"type": "text", "text": "这段媒体在哪个时刻画面与声音同时出现事件？请直接回答选项字母。"
                                     "A) 3.2 秒 B) 5.0 秒 C) 1.0 秒 D) 没有同时出现"},
        ]}]
        attempts = [("flag_true", True), ("flag_false", False)]
        errs = []
        model_inputs = None
        mode = None
        for name, flag in attempts:
            try:
                model_inputs, text = prepare_omni_inputs(processor, conversation,
                                                         use_audio_in_video=flag)
                mode = name
                break
            except Exception as e:
                errs.append((name, type(e).__name__, repr(e)[:300]))
        if model_inputs is None:
            raise RuntimeError(f"prepare_omni_inputs 全部失败: {errs}")
        evidence["processor_mode"] = mode
        evidence["processor_mode_errors"] = errs
        evidence["input_keys"] = list(model_inputs.keys())

        ids = np.array(model_inputs["input_ids"])[0]
        evidence["input_ids_len"] = int(len(ids))
        np.savez_compressed(OUTPUT / "input_ids_b.npz", input_ids=ids)

        tok = processor.tokenizer
        def token_id(name):
            for src in (getattr(processor, name, None), getattr(tok, name, None)):
                if src is None:
                    continue
                v = tok.convert_tokens_to_ids(src) if isinstance(src, str) else src
                if isinstance(v, int) and v >= 0:
                    return v
            return None
        v_id, a_id = token_id("video_token"), token_id("audio_token")
        evidence["media_token_ids"] = {"video": v_id, "audio": a_id}

        if v_id is None or a_id is None:
            evidence["packaging"] = {"status": "未验证", "reason": "媒体 token id 内省失败"}
        else:
            got = runs_of(ids.tolist(), v_id, a_id)
            evidence["backend_media_runs"] = got
            vgt = model_inputs.get("video_grid_thw")
            if vgt is not None:
                g = np.array(vgt)[0]
                grid_t, gh, gw = int(g[0]), int(g[1]), int(g[2])
                merge = getattr(getattr(processor, "video_processor", None), "merge_size", 2) or 2
                tpf = (gh // merge) * (gw // merge)
                n_audio = sum(r["n_tokens"] for r in got if r["modality"] == "audio")
                exp = expected_media_runs(grid_t, n_audio, FPS, tpf, pip, spc)
                evidence["video_grid_thw"] = [grid_t, gh, gw]
                evidence["tokens_per_patch"] = tpf
                evidence["expected_media_runs"] = exp
                match = got == exp
                evidence["packaging"] = {
                    "status": "一致" if match else "差异",
                    "backend_runs": got,
                    "diff": [] if match else [
                        {"i": i, "backend": b, "expected": e}
                        for i, (b, e) in enumerate(zip(got, exp)) if b != e]
                        or [{"note": f"运行数不同: 后端 {len(got)} vs 期望 {len(exp)}"}],
                }
            else:
                evidence["packaging"] = {"status": "未验证", "reason": "video_grid_thw 不可得"}
    except Exception as e:
        import traceback
        hard_error = f"{type(e).__name__}: {repr(e)[:400]}"
        evidence["hard_error"] = hard_error
        evidence["traceback_tail"] = traceback.format_exc()[-1600:]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "report.json"
    report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(p for p in OUTPUT.iterdir() if p.is_file())
    manifest = {"schema": "av_align.p2_01_processor_check_manifest/0-draft",
                "generated": evidence["date"],
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("EVIDENCE:", OUTPUT)
    print(json.dumps(evidence, ensure_ascii=False, indent=2)[:2500])
    return 1 if hard_error else 0


if __name__ == "__main__":
    sys.exit(main())
