from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"

np.random.seed(1)

f = lambda x: np.sin(2 * np.pi * x)
df = lambda x: 2 * np.pi * np.cos(2 * np.pi * x)
SAMPLES = 32  # half the number of samples
NOISE = 0.2

X = np.random.uniform(-1, 1, (SAMPLES, 1))

Y = f(X) + NOISE * np.random.uniform(-1, 1, (SAMPLES, 1))
DY = df(X) + NOISE * np.random.uniform(-1, 1, (SAMPLES, 1))

np.savez(DATA_DIR / f"sobolev_sine.npz", X=X, Y=Y, DY=DY)
