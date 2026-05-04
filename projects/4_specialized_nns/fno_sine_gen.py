from doctest import script_from_examples
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"

rng = np.random.default_rng(2)

# ------------------------- generation settings --------------------------
RESOLUTIONS = [32, 64, 128, 256]

SAMPLES = 128
NOISE = 0.2

shifts = rng.uniform(0, 1, SAMPLES)

f = lambda x, shift: np.sin(2 * np.pi * (x + shift))
df = lambda x, shift: 2 * np.pi * np.cos(2 * np.pi * (x + shift))

# ---------------------------- generate data -----------------------------
for resolution in RESOLUTIONS:
    x = np.linspace(-1, 1, resolution)
    X = np.zeros((SAMPLES, resolution))
    Y = np.zeros((SAMPLES, resolution))
    for i, shift in enumerate(shifts):
        X[i] = f(x, shift) + NOISE * rng.uniform(-1, 1, resolution)
        Y[i] = df(x, shift) + NOISE * rng.uniform(-1, 1, resolution)

    np.savez(DATA_DIR / f"fno_sine_{resolution}.npz", x=x, X=X, Y=Y)
