"""Perception Test 候选错例筛选（research-line 阶段 A 公开候选预筛）。

在 bhqc20 A100 上用 Qwen2.5-Omni-3B（固定 revision，与 P0-01 同源）对 PT valid 集
中"需要音画双证据"的题目跑推理，答案与 GT 不符者标为**候选错例（待研究线确认）**。

- 视频从远端 valid_videos.zip 按 Range 抽取（remotezip，只取需要的文件）；
- 音频从容器经 librosa 抽取后单独传入（4.57 语义，P0-01 已验证）；
- 评分：选项字母解析 vs answer_id；不一致 → 候选。
- 边界：本脚本只产出候选与机制层证据，错例确认权在研究线。
"""

import json
import os
import pathlib
import subprocess
import sys
import time

PIN = "transformers==4.57.6"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
ZIP_URL = "https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip"
OUT = pathlib.Path("/content/pt_screen_evidence") / __import__("datetime").datetime.now().strftime("%Y%m%d")

ITEMS = [
    # (video_id, qid, question, options, answer_id)  GT 字母 = options[answer_id] 首字母
    ("video_10857", 4, "What is the order of the letters at the end?", ["nme", "men", "emn"], 1),
    ("video_10857", 5, "What was the order of the letters at the beginning?", ["men", "enm", "emn"], 0),
    ("video_4956", 3, "What is the order of the letters at the end?", ["fal", "lfa", "afl"], 0),
    ("video_4956", 4, "What was the order of the letters at the beginning?", ["laf", "fal", "afl"], 1),
    ("video_567", 4, "What is the order of the letters at the end?", ["mer", "emr", "rme"], 0),
    ("video_567", 5, "What was the order of the letters at the beginning?", ["rme", "rem", "emr"], 2),
    ("video_7028", 2, "What is the order of the letters at the end?", ["aec", "ace", "cae"], 1),
    ("video_8913", 1, "How many times did the person launch the object on the slanted plane?", ["2", "3", "6"], 1),
    ("video_6260", 1, "How many times did the person launch the object on the slanted plane?", ["4", "6", "3"], 1),
    ("video_9491", 1, "How many times did the person launch objects on the table?", ["7", "3", "2"], 2),
]
NEEDED = sorted({i[0] for i in ITEMS})


def gt_letter(options, answer_id):
    return options[answer_id].strip()[0].upper()


def parse_answer(text, options):
    """从模型回答解析选项字母：优先单字母行/前缀，否则匹配选项原文。"""
    t = text.strip()
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        if t.upper().startswith(letter) and (len(t) == 1 or not t[1].isalpha()):
            return letter, t
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        if opt.lower()[:20] in t.lower():
            return letter, t
    return None, t


def main():
    import cv2
    import librosa
    import torch

    # 内核持久化防御：清理上次运行残留的 GPU 大张量（实测 38 万 token 输入会占满 40GiB）
    import gc
    for stale in ("inputs", "replay_inputs", "out1", "out2", "out_replay", "out_pass",
                  "out_single", "out_both", "model_inputs", "pos_native", "LOGITS_ROW"):
        if stale in globals():
            try:
                del globals()[stale]
            except Exception:
                pass
    gc.collect()
    torch.cuda.empty_cache()
    import transformers
    if transformers.__version__ != "4.57.6":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", PIN], check=True)
        import importlib
        importlib.reload(transformers)
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from huggingface_hub import snapshot_download

    ev = {"schema": "av_align.pt_screen_evidence/0-draft",
          "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "model": MODEL_ID, "revision": REVISION,
          "transformers": transformers.__version__,
          "fps": 2.0, "note": "候选筛选：答案≠GT 者为候选错例（待研究线确认）"}
    items_out = []

    model_path = snapshot_download(MODEL_ID, revision=REVISION)
    processor = AutoProcessor.from_pretrained(model_path)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to("cuda")
    thinker.eval()

    # 远端 zip 抽取需要的视频
    import remotezip
    work = "/content/pt_media"
    os.makedirs(work, exist_ok=True)
    with remotezip.RemoteZip(ZIP_URL) as z:
        names = [n for n in z.namelist() if any(f"{v}." in n for v in NEEDED)]
        print("zip 内匹配文件:", names)
        for n in names:
            if not (pathlib.Path(work) / os.path.basename(n)).exists():
                z.extract(n, work)
    vid_paths = {}
    for v in NEEDED:
        import glob
        hits = glob.glob(f"{work}/**/{v}.*", recursive=True)
        assert hits, f"视频缺失: {v}"
        vid_paths[v] = hits[0]
    print("视频就绪:", vid_paths)

    for vid, qid, question, options, answer_id in ITEMS:
        rec = {"video": vid, "qid": qid, "question": question,
               "options": options, "gt_letter": gt_letter(options, answer_id)}
        try:
            path = vid_paths[vid]
            audio, _ = librosa.load(path, sr=16000, mono=True)
            # 4.57 语义（P0-01 实测）：conversation 只放 video part，音频走 audio= kwarg；
            # 若同时放 audio part，模板双占位符会二次消费音频迭代器 → StopIteration
            conversation = [{"role": "user", "content": [
                {"type": "video", "video": path},
                {"type": "text", "text": question + " Answer with the option's letter from A, B, C directly."},
            ]}]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            # token 预算：720p 帧 × fps 受 max_pixels 约束；超 30k 降 fps 重试
            def build_inputs(fps, max_pixels):
                kw = dict(videos=[path], audio=[audio.astype("float32")],
                          use_audio_in_video=True, fps=fps,
                          videos_kwargs={"fps": fps, "max_pixels": max_pixels})
                return processor(text=[text], **kw, return_tensors="pt").to("cuda"), kw
            inputs, mode = None, None
            perrs = []
            for fps, mp in ((2.0, 768 * 28 * 28), (1.0, 448 * 448)):
                try:
                    inputs, kw_used = build_inputs(fps, mp)
                    n_tok = inputs["input_ids"].shape[1]
                    if n_tok > 30000:
                        perrs.append((f"fps={fps}", f"token 超预算 {n_tok}"))
                        continue
                    mode = f"fps={fps} max_pixels={mp} tokens={n_tok}"
                    break
                except Exception as e:
                    perrs.append((f"fps={fps} mp={mp}", repr(e)[:200]))
            if inputs is None:
                rec["error"] = f"processor 失败: {perrs}"
                items_out.append(rec)
                continue
            with torch.no_grad():
                gen = thinker.generate(**inputs, use_cache=False, max_new_tokens=32,
                                       do_sample=False)
            gen_ids = gen[:, inputs["input_ids"].size(1):]
            raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            letter, _ = parse_answer(raw, options)
            rec.update({"processor_mode": mode, "model_raw": raw[:120],
                        "parsed_letter": letter,
                        "correct": (letter == rec["gt_letter"]) if letter else None})
            if letter is None:
                rec["parse_note"] = "无法解析选项字母——记为未判定"
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {repr(e)[:300]}"
        items_out.append(rec)
        print(json.dumps(rec, ensure_ascii=False)[:220])

    n_judged = sum(1 for r in items_out if r.get("correct") is not None)
    n_wrong = sum(1 for r in items_out if r.get("correct") is False)
    ev["items"] = items_out
    ev["summary"] = {"total": len(items_out), "judged": n_judged,
                     "correct": sum(1 for r in items_out if r.get("correct")),
                     "wrong_candidates": n_wrong,
                     "unparsed_or_error": len(items_out) - n_judged}
    evidence_path = OUT / "report.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in items_out:
        mark = "✗候选" if r.get("correct") is False else ("✓" if r.get("correct") else "?")
        print(f"[{mark}] {r['video']} Q{r['qid']} GT={r['gt_letter']} 模型={r.get('parsed_letter')} raw={r.get('model_raw', r.get('error', ''))[:60]}")
    print(f"候选错例（待研究线确认）: {n_wrong}/{n_judged} | 报告: {evidence_path}")


if __name__ == "__main__":
    main()
