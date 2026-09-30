import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cuwave.wave import simulate

from helper import FWI, load_cmap

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--theme", choices=["light", "dark"], default="light")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
SHOT = 1  # of the four sources, left to right
FRAMES = 150
CLIP = 0.05  # colour range as a fraction of the largest pressure, the source dominates
FPS = 20
cmap = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")
name = f"neuralfwi_wave_{args.theme}"
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames" / name

# ------------------------------------- simulation ------------------------------------
fwi = FWI()
truth = fwi.circles()
_, snapshots = simulate(
    fwi.sim, fwi.sources[SHOT], truth, record_every=fwi.sim.N // FRAMES
)
snapshots = np.asarray(snapshots)[:, 1:-1, 1:-1]
voids = truth[1:-1, 1:-1].get() < 0.5
limit = CLIP * np.abs(snapshots).max()
extent = [0, 1, 0, 1]
sensors, sources = fwi.sensor_coords, fwi.source_coords

# ----------------------------------- postprocessing ----------------------------------
ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
for frame in ANIMATION_DIR.glob("frame_*.jpg"):  # a finer grid writes fewer frames
    frame.unlink()
hole_cmap = "gray_r" if args.theme == "light" else "gray"  # black or white holes
overlay = np.ma.masked_where(~voids.T, voids.T)
for i, u in enumerate(snapshots):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=128)
    ax.imshow(
        u.T, origin="lower", extent=extent, cmap=cmap, vmin=-limit, vmax=limit
    )
    ax.imshow(overlay, origin="lower", extent=extent, cmap=hole_cmap, vmin=0, vmax=1)
    # markers centred on the top edge, so only their lower half shows
    marker = dict(clip_on=False, zorder=3, mew=0)
    ax.plot(*sources.T, "o", ms=10, mfc="w", **marker)
    ax.plot(*sources[SHOT], "o", ms=10, mfc="r", **marker)
    ax.plot(*sensors.T, "o", ms=5, mfc="k", **marker)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(ANIMATION_DIR / f"frame_{i}.jpg")
    plt.close()

frames = str(ANIMATION_DIR / "frame_%d.jpg")
video = str(RESULTS_DIR / f"{name}.mp4")
ffmpeg = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS)]
subprocess.run(ffmpeg + ["-i", frames, "-pix_fmt", "yuv420p", video], check=True)
