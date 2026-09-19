import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import numpy as np
from cuwave.wave import simulate
from PIL import Image

from helper import MIN_INDICATOR, build, indicator

BASE_DIR = Path(__file__).parent
ANIMATION_DIR = (BASE_DIR / "../../results/animations/animation_frames").resolve()

# -------------------------------------- settings -------------------------------------
T = 1e-5

# animation
SAVE_EVERY = 2
SCALE = 5e-6  # fixed color range, so the frames share one scale
VOID_GREY = 178  # Greys_r at 0.7

# --------------------------------------- setup ---------------------------------------
sim, source, indicator_padded = build(T)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
_, snaps = simulate(sim, source, indicator_padded, record_every=SAVE_EVERY)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / sim.N * 1e3:.4f} ms/step)")

# ----------------------------------- animate export ----------------------------------
frame_dir = ANIMATION_DIR / "bhai_waves"
frame_dir.mkdir(parents=True, exist_ok=True)
void = indicator == MIN_INDICATOR

for i, snap in enumerate(snaps):
    # normalize wave field to [0, 1] and apply colormap
    normalized = np.clip((snap + SCALE) / (2 * SCALE), 0, 1)
    rgb = (cmr.fusion(normalized)[:, :, :3] * 255).astype(np.uint8)
    rgb[void] = VOID_GREY
    Image.fromarray(rgb, mode="RGB").save(frame_dir / f"frame_{i}.jpg", quality=85)
print(f"saved {len(snaps)} frames to {frame_dir}")
