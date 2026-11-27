"""Generate two original reference clips (no downloads). Run from a repository checkout."""

import argparse
import pathlib
import subprocess

import numpy as np

from p0_03_replay import ffmpeg_exe


def make_pair(directory):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # One 2 s, 12 fps visual: a light square appears at 1.00 s for three frames.
    visual = ("color=c=0x182030:s=96x96:r=12:d=2,"
              "drawbox=x=24:y=24:w=48:h=48:color=white:t=fill:enable='between(n,12,14)'")
    before, after = (directory / f"high-{side}.mp4" for side in ("before", "after"))
    sr = 48000
    t = np.arange(sr // 8) / sr
    for path, tones in [(before, (880, 440)), (after, (440, 880))]:
        pcm = np.zeros(2 * sr, dtype="<f4")
        for start, freq in zip((sr // 2, 3 * sr // 2), tones):
            pcm[start:start + len(t)] = 0.6 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
        # Copy the first clip's video bitstream into the second; never re-render it.
        video_in = ["-f", "lavfi", "-i", visual] if path == before else ["-i", str(before)]
        subprocess.run([ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-n", *video_in,
                        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4" if path == before else "copy",
                        "-c:a", "aac", "-b:a", "96k", str(path)], input=pcm.tobytes(), check=True)
    return before, after


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="destination; existing clips are not overwritten")
    for clip in make_pair(parser.parse_args().directory):
        print(clip)
