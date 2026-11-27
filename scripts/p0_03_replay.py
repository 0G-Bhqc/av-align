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
      --env QUESTION='研究线确认的完整问题（含选项文本）' \
      --env OPTIONS='A|B|C|D' \
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
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
PIN = "transformers==4.57.6"

MEDIA = os.environ.get("MEDIA", "/content/media/input.mp4")
QUESTION = os.environ.get("QUESTION", "")
OPTIONS_RAW = os.environ.get("OPTIONS", "A|B|C|D")
DELTA_UNITS = int(os.environ.get("DELTA_UNITS", "25"))
OUT = pathlib.Path(os.environ.get("OUTDIR", "/content/p0_03_evidence")) / datetime.datetime.now().strftime("%Y%m%d")


SAMPLE_FPS = 4.0
AUDIO_RATE = 16000


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as e:
        raise RuntimeError("需要 PATH 上的 ffmpeg 或 imageio-ffmpeg") from e
    return get_ffmpeg_exe()


def decode_media(media):
    """Read the first video/audio streams on the container clock; no intermediate files."""
    path = pathlib.Path(media).resolve(strict=True)
    cmd = [ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-xerror", "-copyts",
           "-start_at_zero", "-i", str(path)]

    def stream(selector, *args):
        p = subprocess.run([*cmd, "-map", selector, *args, "pipe:1"], capture_output=True)
        if p.returncode or not p.stdout:
            raise ValueError(f"{path}: cannot decode {selector}: {p.stderr.decode(errors='replace')[-500:]}")
        return p.stdout

    # first_pts=0 preserves a delayed track's initial gap, without inventing a missing track.
    pcm = stream("0:a:0", "-af", f"aresample={AUDIO_RATE}:async=1:first_pts=0",
                 "-ac", "1", "-f", "f32le")
    audio = np.frombuffer(pcm, dtype="<f4").astype(np.float32, copy=True)
    # Sample by PTS, not by relabelling every source frame as 4 fps. FFmpeg's P6
    # framing supplies the post-rotation dimensions without ffprobe or a second decoder.
    ppm = io.BytesIO(stream("0:v:0", "-vf", f"fps={SAMPLE_FPS}:start_time=0:round=up",
                            "-pix_fmt", "rgb24", "-c:v", "ppm", "-f", "image2pipe"))
    frames = []
    while magic := ppm.readline():
        if magic != b"P6\n":
            raise ValueError(f"{path}: invalid FFmpeg PPM frame")
        w, h = map(int, ppm.readline().split())
        if ppm.readline() != b"255\n":
            raise ValueError(f"{path}: expected 8-bit RGB")
        frames.append(np.frombuffer(ppm.read(h * w * 3), np.uint8).reshape(h, w, 3))
    return np.stack(frames).transpose(0, 3, 1, 2), audio


def decoded_summary(video, audio):
    rgb = np.ascontiguousarray(video.transpose(0, 2, 3, 1))
    return {"frames": len(video), "fps": SAMPLE_FPS, "audio_samples": len(audio),
            "sample_rate": AUDIO_RATE, "rgb24_sha256": hashlib.sha256(rgb).hexdigest(),
            "pcm_f32le_sha256": hashlib.sha256(audio.astype("<f4", copy=False)).hexdigest()}


def processor_inputs(processor, media, question, video, audio):
    # 4.57.6: the video part consumes the separately supplied audio once.
    conversation = [{"role": "user", "content": [
        {"type": "video", "video": str(media)}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    return processor(text=[text], videos=[video], audio=[audio], use_audio_in_video=True,
                     videos_kwargs={"fps": SAMPLE_FPS}, return_tensors="pt")


def option_ids(tokenizer, raw):
    result = {}
    for option in raw.split("|"):
        label = option.partition(")")[0].strip()
        if len(label) != 1 or not "A" <= label <= "Z" or label in result:
            raise ValueError(f"expected unique option letters, got {option!r}")
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in (label, " " + label)]
        result[label] = list(dict.fromkeys(seq[0] for seq in ids
                                           if len(seq) == 1 and seq[0] != tokenizer.unk_token_id))
        if not result[label]:
            raise ValueError(f"no single-token spelling for {label}")
    return result


def score_options(row, options):
    # Best next-token logit across valid spellings (e.g. 'A' and ' A').
    scores = {label: row[ids].max().item() for label, ids in options.items()}
    return {"option_logits": scores, "argmax": max(scores, key=scores.get)}


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

        # ② 先解码验证同源音画；无音轨/坏媒体在加载权重前明确失败。
        video_tc, audio_stream = decode_media(MEDIA)
        evidence["decoded"] = decoded_summary(video_tc, audio_stream)

        # ③ 固定 revision 权重（缓存命中即快）
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(MODEL_ID, revision=REVISION)

        # ④ 模板 + processor（use_audio_in_video=True + 音频单独传入）
        from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path)
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to("cuda")
        thinker.eval()
        inputs = processor_inputs(processor, MEDIA, QUESTION, video_tc, audio_stream).to("cuda")
        evidence["processor_mode"] = "video_part+audio_kwarg"
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
        OPTIONS = option_ids(processor.tokenizer, OPTIONS_RAW)
        replay_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        REPLAY_FP = {k: t_sha(v) for k, v in replay_inputs.items()}
        discrete_same = {k: INPUTS_FP[k] == REPLAY_FP[k] for k in INPUTS_FP}
        with torch.no_grad():
            out_replay = thinker(**replay_inputs, use_cache=False)
        DIFF_REPLAY = (out_replay.logits[0, FINAL_IDX].float() - LOGITS_ROW).abs().max().item()
        evidence["native"] = {"jitter": JITTER, "option_ids": OPTIONS, **score_options(LOGITS_ROW, OPTIONS)}
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
            return {"logits_max_abs_diff": (r - LOGITS_ROW).abs().max().item(), **score_options(r, OPTIONS)}
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

    out = write_evidence(OUT, evidence)
    print("EVIDENCE:", out)
    print(json.dumps(evidence, ensure_ascii=False, indent=2)[:2200])
    return 1 if hard_error else 0


def write_evidence(day_dir, evidence):
    # The same day may include both clips (or a failed retry). Never re-hash an old manifest.
    day_dir.mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(tempfile.mkdtemp(prefix="run-", dir=day_dir))
    report = out / "report.json"
    report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"schema": "av_align.p0_03_replay_manifest/0-draft", "generated": evidence["date"],
                "files": {report.name: hashlib.sha256(report.read_bytes()).hexdigest()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--inspect-media"]:
        if not sys.argv[2:]:
            sys.exit("usage: p0_03_replay.py --inspect-media FILE [FILE ...]")
        for path in sys.argv[2:]:
            print(json.dumps({"media": path, **decoded_summary(*decode_media(path))}, ensure_ascii=False))
    else:
        sys.exit(main())
