"""
Convert a folder of animation frames (PNG/JPG) to a video file.

Uses ffmpeg (H.264 + AAC-less) for smallest file size at good quality.
Adjust `crf` (0=lossless, 51=worst; 18–23 is visually lossless range).

Usage:
    python frames_to_animation.py --frames_dir ../../results/animations/animation_frames \
                                   --output     ../../results/animations/balls.mp4 \
                                   --pattern    "balls_%d.jpg" \
                                   --fps        30 \
                                   --crf        20
"""
import argparse
import subprocess
from pathlib import Path


def frames_to_video(frames_dir, output, pattern="frame_%d.jpg", fps=30, crf=20):
    frames_dir = Path(frames_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    input_path = frames_dir / pattern

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(input_path),
        "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",  # ensures even dimensions
        "-c:v", "libx264",
        "-preset", "slow",        # better compression, same quality
        "-crf", str(crf),         # quality: lower = better (18-23 recommended)
        "-pix_fmt", "yuv420p",    # broad compatibility
        "-colorspace", "bt709",
        "-color_range", "tv",
        "-movflags", "+faststart", # web streaming: moov atom at front
        str(output),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_mb = output.stat().st_size / 1e6
    print(f"Saved {output}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert frames to video via ffmpeg.")
    parser.add_argument("--frames_dir", required=True,  help="Directory with frame images")
    parser.add_argument("--output",     required=True,  help="Output video path (.mp4)")
    parser.add_argument("--pattern",    default="frame_%d.jpg", help="Frame filename pattern (ffmpeg style)")
    parser.add_argument("--fps",        default=30,  type=int,   help="Frames per second")
    parser.add_argument("--crf",        default=20,  type=int,   help="Quality (0=lossless, 51=worst; 18-23 recommended)")
    args = parser.parse_args()

    frames_to_video(args.frames_dir, args.output, args.pattern, args.fps, args.crf)
