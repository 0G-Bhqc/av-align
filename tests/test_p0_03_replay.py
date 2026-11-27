"""Offline replay boundary tests: real containers and PCM; no model or processor downloads."""

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def replay():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(SCRIPTS))
        import p0_03_replay
    return p0_03_replay


@pytest.fixture(scope="module")
def ffmpeg(replay):
    try:
        return replay.ffmpeg_exe()
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def pair(tmp_path_factory, ffmpeg):
    out = tmp_path_factory.mktemp("original clips")
    subprocess.run([sys.executable, str(SCRIPTS / "make_replay_pair.py"), str(out)], check=True)
    return out / "high-before.mp4", out / "high-after.mp4"


def pitch(pcm, start, sr=16000):
    segment = pcm[round(start * sr):round((start + 0.125) * sr)]
    assert np.max(np.abs(segment)) > 0.3, "missing audible cue"
    return np.fft.rfftfreq(len(segment), 1 / sr)[abs(np.fft.rfft(segment)).argmax()]


def test_identical_source_frames_and_reversed_real_cues(pair, ffmpeg, replay):
    # Independent check on *all* 24 source frames, not just the replay's 8 sampled frames.
    def full_rgb(path):
        return subprocess.check_output([ffmpeg, "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
                                        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])

    source = full_rgb(pair[0])
    assert len(source) == 24 * 96 * 96 * 3 and source == full_rgb(pair[1])
    for path, expected in zip(pair, [(880, 440), (440, 880)]):
        video, audio = replay.decode_media(path)
        assert video.shape == (8, 3, 96, 96) and video.dtype == np.uint8
        # The flash is still at 1 s, not 3 s (all source frames mislabeled as 4 fps).
        assert np.flatnonzero(video[:, 0, 48, 48] > 200).tolist() == [4]
        assert audio.dtype == np.float32 and audio.ndim == 1
        # Some AAC decoders retain up to one trailing 48 kHz codec frame.
        assert 32000 <= len(audio) <= 32000 + 342
        assert (pitch(audio, 0.5), pitch(audio, 1.5)) == pytest.approx(expected, abs=8)
        assert np.max(np.abs(audio[16000:18000])) < 0.01


def test_same_path_replaced_a_b_a(pair, replay, tmp_path):
    current = tmp_path / "input.mp4"
    summaries = []
    for source in (*pair, pair[0]):
        shutil.copyfile(source, current)
        summaries.append(replay.decoded_summary(*replay.decode_media(current)))
    first, second, third = summaries
    assert first == third
    assert first["rgb24_sha256"] == second["rgb24_sha256"]
    assert first["pcm_f32le_sha256"] != second["pcm_f32le_sha256"]
    assert list(tmp_path.iterdir()) == [current]


def test_missing_audio_is_not_silence_or_a_cached_track(pair, ffmpeg, replay, tmp_path):
    current = tmp_path / "input.mp4"
    shutil.copyfile(pair[0], current)
    replay.decode_media(current)
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(pair[0]), "-an", "-c:v", "copy",
                    str(current)], check=True)
    with pytest.raises(ValueError, match="0:a:0"):
        replay.decode_media(current)


def test_real_silent_track_remains_silent(pair, ffmpeg, replay, tmp_path):
    path = tmp_path / "quiet.mkv"
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(pair[0]), "-af", "volume=0",
                    "-c:v", "copy", "-c:a", "pcm_s16le", str(path)], check=True)
    _, audio = replay.decode_media(path)
    assert len(audio) >= 32000 and not np.any(audio)


def test_delayed_audio_stream_stays_on_video_clock(pair, ffmpeg, replay, tmp_path):
    path = tmp_path / "delayed.mp4"
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(pair[0]), "-itsoffset", "0.25",
                    "-i", str(pair[0]), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(path)], check=True)
    video, audio = replay.decode_media(path)
    assert len(video) == 8
    assert np.max(np.abs(audio[:int(0.70 * 16000)])) < 0.01
    assert pitch(audio, 0.75) == pytest.approx(880, abs=8)
    assert pitch(audio, 1.75) == pytest.approx(440, abs=8)


def test_processor_has_one_video_part_and_receives_pcm_once(replay):
    video = np.zeros((8, 3, 2, 2), dtype=np.uint8)
    audio = np.arange(32000, dtype=np.float32)

    class Spy:
        def apply_chat_template(self, conversation, **kwargs):
            assert conversation == [{"role": "user", "content": [{"type": "video", "video": "source.mp4"},
                                                                 {"type": "text", "text": "Q?"}]}]
            assert kwargs == {"add_generation_prompt": True, "tokenize": False}
            return "rendered"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["rendered"] and kwargs["return_tensors"] == "pt"
            assert len(kwargs["videos"]) == 1 and kwargs["videos"][0] is video
            assert len(kwargs["audio"]) == 1 and kwargs["audio"][0] is audio
            assert kwargs["use_audio_in_video"] is True and kwargs["videos_kwargs"] == {"fps": 4.0}
            return "inputs"

    assert replay.processor_inputs(Spy(), "source.mp4", "Q?", video, audio) == "inputs"


def test_same_day_reports_are_not_overwritten_or_cross_hashed(tmp_path, replay):
    (tmp_path / "manifest.json").write_bytes(b"old manifest")
    outputs = [replay.write_evidence(tmp_path, {"date": "same day", "media": name}) for name in ("A", "B")]
    assert outputs[0] != outputs[1]
    for out, expected in zip(outputs, ("A", "B")):
        report = (out / "report.json").read_bytes()
        assert json.loads(report)["media"] == expected
        manifest = json.loads((out / "manifest.json").read_bytes())
        assert manifest["files"] == {"report.json": hashlib.sha256(report).hexdigest()}


def test_scalar_option_scores_with_two_valid_spellings(replay):
    class Tokenizer:
        unk_token_id = 0

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return {"A": [1], " A": [2], "B": [3], " B": [3], "C": [0], " C": [4, 5]}[text]

    ids = replay.option_ids(Tokenizer(), "A) before | B) after")
    assert ids == {"A": [1, 2], "B": [3]}
    row = np.array([20., -2., 0.75, 0.5, 1.])
    with pytest.raises(ValueError):  # the former row[ids].item() required exactly one element
        row[ids["A"]].item()
    assert replay.score_options(row, ids) == {"argmax": "A", "option_logits": {"A": 0.75, "B": 0.5}}
    with pytest.raises(ValueError, match="no single-token"):
        replay.option_ids(Tokenizer(), "C")
