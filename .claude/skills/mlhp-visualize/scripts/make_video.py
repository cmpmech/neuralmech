#!/usr/bin/env python
"""Render a time series of mlhp VTU/PVTU files to an MP4 (headless vtk + ffmpeg).

Computes ONE global colour range across all frames (so the colorbar/colours
don't flicker), renders each frame, then encodes with ffmpeg. ffmpeg is
discovered (PATH, then $MLHP_FFMPEG, then --ffmpeg); never hardcoded.

Examples
--------
  # 2D flow series, colour by velocity magnitude:
  python make_video.py "outputs/flow_*.pvtu" --field mag:Solution --out flow.mp4

  # 3D deformation series with the deformed mesh overlaid:
  python make_video.py "outputs/solid_*.pvtu" --field VonMisesStress --scale 1e-6 \
      --warp Displacement --view iso --edges-pattern "outputs/edges_*.pvtu" \
      --bar-title "vM [MPa]" --fps 4 --out deform.mp4

Frame files are sorted by the integer in their name (so _10 follows _9, not _1).
Give patterns that match only their own files: a shared prefix collides (e.g.
"foo_*.pvtu" also catches "foo_edges_*.pvtu") -- use "foo_[0-9]*.pvtu" for frames.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import mlhp_visualize as mlhpviz


def numsort(paths):
    def key(p):
        m = re.search(r"(\d+)\.[^.]+$", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=key)


def find_ffmpeg(explicit):
    cand = explicit or os.environ.get("MLHP_FFMPEG") or shutil.which("ffmpeg")
    if not cand or not (os.path.exists(cand) or shutil.which(cand)):
        sys.exit("ffmpeg not found. Put it on PATH, set $MLHP_FFMPEG, or pass --ffmpeg PATH.")
    return cand


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern", help="glob for the frame files, quoted, e.g. 'outputs/flow_*.pvtu'")
    p.add_argument("--out", default="video.mp4")
    p.add_argument("--field", required=True, help="NAME | mag:NAME | comp:NAME:i")
    p.add_argument("--view", default="2d",
                   choices=["2d", "iso", "+x", "-x", "+y", "-y", "+z", "-z"])
    p.add_argument("--warp", default=None, metavar="VECNAME")
    p.add_argument("--warp-scale", type=float, default=1.0)
    p.add_argument("--range", default="p99",
                   help="'auto' | 'pNN' (default p99, robust to cut-cell spikes) | 'LO,HI'")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--cmap", default="turbo", choices=list(mlhpviz._CMAPS))
    p.add_argument("--edges-pattern", default=None, help="glob for matching mesh-edge files")
    p.add_argument("--bar-title", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--size", default=None, help="WxH")
    p.add_argument("--label-fmt", default="%.2f")
    p.add_argument("--ffmpeg", default=None, help="explicit ffmpeg path (else PATH/$MLHP_FFMPEG)")
    p.add_argument("--keep-frames", action="store_true")
    a = p.parse_args()

    files = numsort(glob.glob(a.pattern))
    if not files:
        sys.exit(f"no files match {a.pattern!r}")
    edges = numsort(glob.glob(a.edges_pattern)) if a.edges_pattern else [None] * len(files)
    if a.edges_pattern and len(edges) != len(files):
        sys.exit(f"edge count {len(edges)} != frame count {len(files)}")
    ffmpeg = find_ffmpeg(a.ffmpeg)
    size = tuple(int(x) for x in a.size.lower().split("x")) if a.size else None
    print(f"{len(files)} frames")

    # explicit range stays as given; otherwise compute ONE global range up front
    if "," in a.range:
        lo, hi = a.range.split(","); grange = (float(lo), float(hi))
    else:
        allvals = []
        for f in files:
            d = mlhpviz.read_dataset(f)
            if a.warp:
                d = mlhpviz.warp(d, a.warp, a.warp_scale)
            _, vals = mlhpviz.resolve_field(d, a.field, a.scale)
            allvals.append(vals)
        cat = np.concatenate(allvals)
        grange = mlhpviz.value_range(cat, a.range)
    print(f"global range [{grange[0]:.4g}, {grange[1]:.4g}]")

    tmp = tempfile.mkdtemp(prefix="mlhp_visualize_frames_")
    try:
        for i, (f, e) in enumerate(zip(files, edges)):
            out = os.path.join(tmp, f"frame_{i:04d}.png")
            mlhpviz.render(f, out, a.field, view=a.view, warp_field=a.warp,
                           warp_scale=a.warp_scale, rng=grange, scale=a.scale, cmap=a.cmap,
                           title=a.title, bar_title=a.bar_title, label=None,
                           edges=e, size=size, label_fmt=a.label_fmt)
        cmd = [ffmpeg, "-y", "-framerate", str(a.fps),
               "-i", os.path.join(tmp, "frame_%04d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", a.out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(a.out):
            sys.exit("ffmpeg failed:\n" + r.stderr[-1500:])
        print(f"wrote {a.out}  ({len(files)} frames @ {a.fps} fps)")
        if a.keep_frames:
            print("frames in", tmp)
    finally:
        if not a.keep_frames:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
