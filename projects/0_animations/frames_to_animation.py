"""convert a folder of animation frames (PNG/JPG) to an H.264 video with ffmpeg.

`--crf` trades size for quality (0 lossless, 51 worst; 18-23 is visually lossless).

Usage:
    python frames_to_animation.py --frames_dir ../../results/animations/frames \
        --output ../../results/animations/balls.mp4 --pattern "balls_%d.jpg" --fps 30
"""

import argparse
import subprocess
from pathlib import Path


def frames_to_video(frames_dir, output, pattern="frame_%d.jpg", fps=30, crf=20):
    frames_dir = Path(frames_dir).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    input_path = frames_dir / pattern

    # fmt: off
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(input_path),
        "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",  # ensures even dimensions
        "-c:v", "libx264",
        "-preset", "slow",  # better compression, same quality
        "-crf", str(crf),  # quality: lower = better (0-51, but 18-23 recommended)
        "-pix_fmt", "yuv420p",  # broad compatibility
        "-colorspace", "bt709",
        "-color_range", "tv",
        "-movflags", "+faststart",
        str(output)
    ]
    # fmt: on

    print(f"\nrunning: {' '.join(cmd)}")
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    size_mb = output.stat().st_size / 1e6
    print(f"\nsaved {output}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="frame_%d.jpg")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--crf", default=20, type=int)
    args = parser.parse_args()

    frames_to_video(args.frames_dir, args.output, args.pattern, args.fps, args.crf)
