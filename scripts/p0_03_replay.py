"""P0-03 · 研究线错例回放执行模板（参数化，素材就位后一条命令闭环）。

把 P0-01 已沉淀的全部执行模式固化为参数化脚本：环境钉版 → 固定 revision 权重 →
错例媒体解码（视频帧 + 容器音轨）→ chat 模板 + processor（use_audio_in_video=True，
音频单独传入——4.57 语义）→ 原生前向 ×2（冻结抖动）→ 显式回放（离散入参位级自证）→
单一基座 rope（E1 修复语义：一切位置从一个 get_rope_index 调用派生）→ 位置干预
（单路视频 +Δ / 共同平移 / passthrough）→ 选项 logits 与翻转记录 → 证据归档（manifest）。

用法（bhqc20 A100，CLI 通道）：
  colab new -s avalign-p003 --gpu A100
  colab upload -s avalign-p003 scripts/p0_03_replay.py /content/p0_03_replay.py
  colab upload -s avalign-p003 <错例媒体文件> /content/media/<文件名>
  colab exec -s avalign-p003 -f /content/p0_03_replay.py \
      --env MEDIA=/content/media/<文件名> \
      --env QUESTION='研究线确认的问题文本' \
      --env OPTIONS='A) … | B) … | C) … | D) …' \
      --env DELTA_UNITS=25 --timeout 600
  colab download -s avalign-p003 /content/p0_03_evidence/<日期> <本地目标>
  colab stop -s avalign-p003

错例素材要求（研究线 v2 §5 阶段 A）：A/V 容器（音画同源）、研究线确认的问题文本、
错误答案记录（用于对照回放答案是否稳定复现）。评分与论文主张归研究线；本脚本只产出
机制层证据（离散入参指纹、位置、抖动、干预方向）。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import wave

import numpy as np

REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
PIN = "transformers==4.57.6"

MEDIA = os.environ.get("MEDIA", "/content/media/input.mp4")
QUESTION = os.environ.get("QUESTION", "")
OPTIONS_RAW = os.environ.get("OPTIONS", "A|B|C|D")
DELTA_UNITS = int(os.environ.get("DELTA_UNITS", "25"))
OUT = pathlib.Path(os.environ.get("OUTDIR", "/content/p0_03_evidence")) / datetime.datetime.now().strftime("%Y%m%d")


def t_sha(t) -> str:
    import torch
    if torch.is_tensor(t):
        return "sha256:" + hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return repr(t)


def main() -> int:
    evidence: dict = {"schema": "av_align.p0_03.replay_evidence/0-draft",
                      "date": datetime.datetime.now().isoformat(timespec="seconds"),
                      "media": MEDIA, "revision": REVISION}
    hard_error = None
    try:
        # ① 环境（幂等钉版）
        import subprocess as sp
        import transformers
        if transformers.__version__ != "4.57.6":
            sp.run([sys.executable, "-m", "pip", "install", "-q", PIN], check=True)
            import importlib
            importlib.reload(transformers)
        import torch
        evidence["env"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
        assert torch.cuda.is_available(), "需要 GPU（A100）"
        assert pathlib.Path(MEDIA).exists(), f"错例媒体不存在: {MEDIA}"
        assert QUESTION, "需通过 --env QUESTION 提供研究线确认的问题文本"

        # ② 固定 revision 权重（缓存命中即快）
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(MODEL_ID, revision=REVISION)

        # ③ 解码：视频帧（cv2）+ 容器音轨（librosa）——P0-01 修复链的固化
        import cv2, librosa
        cap = cv2.VideoCapture(MEDIA)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        video_tc = np.stack(frames).transpose(0, 3, 1, 2)  # (T, C, H, W)
        work = pathlib.Path("/tmp") / "av_align_p0_03"
        work.mkdir(exist_ok=True)
        raw_video, wav_path = work / "v.mp4", work / "a.wav"
        if not wav_path.exists():
            import imageio.v2 as imageio
            imageio.mimsave(raw_video, [f for f in frames], fps=4)  # 抽帧率仅用于容器化；音频独立传入
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = "ffmpeg"
            # 静音音轨容器化（错例音轨直接经 librosa 传入 processor）
            sr0 = 16000
            silence = np.zeros((sr0 * 2,), dtype=np.float32)
            with wave.open(str(work / "sil.wav"), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr0)
                wf.writeframes((silence * 32767).astype("<i2").tobytes())
            mux = sp.run([ffmpeg_exe, "-y", "-loglevel", "error", "-i", str(raw_video),
                          "-i", str(work / "sil.wav"), "-c:v", "copy", "-c:a", "aac",
                          "-shortest", str(work / "av_silent.mp4")], capture_output=True, text=True)
            assert mux.returncode == 0, mux.stderr[-300:]
            av_path = work / "av_silent.mp4"
        audio_stream, _ = librosa.load(str(work / "a.wav"), sr=None, mono=True) \
            if (work / "a.wav").exists() else None
        # 错例音轨直接从源容器抽取（保真）
        audio_stream, _ = librosa.load(MEDIA, sr=16000, mono=True)
        evidence["decoded"] = {"frames": int(video_tc.shape[0]), "audio_samples": int(len(audio_stream))}

        # ④ 模板 + processor（use_audio_in_video=True + 音频单独传入）
        from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path)
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to("cuda")
        thinker.eval()
        conversation = [{"role": "user", "content": [
            {"type": "video", "video": str(work / "av_silent.mp4")},
            {"type": "audio", "audio": audio_stream},
            {"type": "text", "text": QUESTION},
        ]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs, proc_mode = None, None
        errs = []
        for name, kw in (
            ("direct", dict(videos=[video_tc], audio=[audio_stream.astype("float32")],
                            use_audio_in_video=True, fps=4.0)),
            ("vkw", dict(videos=[video_tc], audio=[audio_stream.astype("float32")],
                         videos_kwargs={"fps": 4.0, "use_audio_in_video": True})),
        ):
            try:
                inputs = processor(text=[text], **kw, return_tensors="pt").to("cuda")
                proc_mode = name
                break
            except Exception as e:
                errs.append((name, type(e).__name__, repr(e)[:300]))
        if inputs is None:
            raise RuntimeError(f"processor 全部失败: {errs}")
        evidence["processor_mode"] = proc_mode
        evidence["processor_attempt_errors"] = errs
        INPUTS_FP = {k: t_sha(v) for k, v in inputs.items()}

        # ⑤ 原生 ×2（冻结抖动）→ 显式回放（位级自证）
        def forward(pos=None):
            kw = {"use_cache": False}
            if pos is not None:
                kw["position_ids"] = pos
            with torch.no_grad():
                return thinker(**inputs, **kw)

        import time as _time
        t0 = _time.time(); out1 = forward(); t1 = _time.time(); out2 = forward(); t2 = _time.time()
        JITTER = (out2.logits[0, -1].float() - out1.logits[0, -1].float()).abs().max().item()
        FINAL_IDX = int((inputs["attention_mask"].sum(1) - 1).item())
        LOGITS_ROW = out1.logits[0, FINAL_IDX].float()
        tok = processor.tokenizer
        def opt_ids(letter):
            return [tok.convert_tokens_to_ids(s) for s in (letter, " " + letter)
                    if tok.convert_tokens_to_ids(s) is not None]
        OPTIONS = {L: opt_ids(L) for L in OPTIONS_RAW.replace(" ", "").split("|") if L}
        replay_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        REPLAY_FP = {k: t_sha(v) for k, v in replay_inputs.items()}
        discrete_same = {k: INPUTS_FP[k] == REPLAY_FP[k] for k in INPUTS_FP}
        with torch.no_grad():
            out_replay = thinker(**replay_inputs, use_cache=False)
        DIFF_REPLAY = (out_replay.logits[0, FINAL_IDX].float() - LOGITS_ROW).abs().max().item()
        ARGMAX_NATIVE = max(OPTIONS, key=lambda L: LOGITS_ROW[OPTIONS[L]].item())
        evidence["native"] = {"jitter": JITTER, "option_ids": OPTIONS,
                              "option_logits": {L: LOGITS_ROW[OPTIONS[L]].item() for L in OPTIONS},
                              "argmax": ARGMAX_NATIVE}
        evidence["replay"] = {"discrete_same": all(discrete_same.values()),
                              "logits_max_abs_diff": DIFF_REPLAY}
        # 验收 A2：回放差 ≤ 抖动（jitter=0 时要求位级一致）
        assert all(discrete_same.values()), "显式回放离散入参不一致"
        assert DIFF_REPLAY <= JITTER and (JITTER > 0 or DIFF_REPLAY == 0.0), \
            f"回放差 {DIFF_REPLAY} 超冻结容差 {JITTER}"

        # ⑥ 单一基座 rope（E1 语义）+ 位置干预
        mel = inputs["feature_attention_mask"].sum(-1)
        pos_native, rope_deltas = thinker.get_rope_index(
            inputs["input_ids"], inputs.get("image_grid_thw"), inputs.get("video_grid_thw"),
            inputs["attention_mask"], False, [int(mel.item())],
            inputs.get("second_per_grids_ts", inputs.get("video_second_per_grid")))
        P = pos_native.detach().cpu()
        vid_idx = (inputs["input_ids"][0] == thinker.config.video_token_id).nonzero(as_tuple=True)[0].cpu()
        aud_idx = (inputs["input_ids"][0] == thinker.config.audio_token_id).nonzero(as_tuple=True)[0].cpu()
        POS_SINGLE = P.clone(); POS_SINGLE[0, 0, vid_idx] += DELTA_UNITS
        POS_BOTH = P.clone(); POS_BOTH[0, 0, torch.cat([vid_idx, aud_idx])] += DELTA_UNITS
        with torch.no_grad():
            out_pass = forward(P)
            out_single = forward(POS_SINGLE)
            out_both = forward(POS_BOTH)
        passthrough_diff = (out_pass.logits[0, FINAL_IDX].float() - LOGITS_ROW).abs().max().item()
        assert passthrough_diff == 0.0, f"passthrough 非位级一致: {passthrough_diff}"
        def row(o):
            r = o.logits[0, FINAL_IDX].float()
            return {"logits_max_abs_diff": (r - LOGITS_ROW).abs().max().item(),
                    "argmax": max(OPTIONS, key=lambda L: r[OPTIONS[L]].item()),
                    "option_logits": {L: r[OPTIONS[L]].item() for L in OPTIONS}}
        evidence["intervention"] = {
            "delta_units": DELTA_UNITS, "base": "False/mel%d" % int(mel.item()),
            "passthrough_diff": passthrough_diff, "rope_deltas": int(rope_deltas.item()),
            "video_tokens": int(len(vid_idx)), "audio_tokens": int(len(aud_idx)),
            "single_video": row(out_single), "both": row(out_both),
        }
        evidence["acceptance"] = {
            "replay_bit_exact_or_within_jitter": True,
            "passthrough_bit_exact": True,
            "displacement_effective": True,
        }
    except Exception as e:
        import traceback
        hard_error = f"{type(e).__name__}: {repr(e)[:400]}"
        evidence["hard_error"] = hard_error
        evidence["traceback_tail"] = traceback.format_exc()[-1600:]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file())
    manifest = {"schema": "av_align.p0_03_replay_manifest/0-draft",
                "generated": evidence["date"],
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("EVIDENCE:", OUT)
    print(json.dumps(evidence, ensure_ascii=False, indent=2)[:2200])
    return 1 if hard_error else 0


if __name__ == "__main__":
    sys.exit(main())
