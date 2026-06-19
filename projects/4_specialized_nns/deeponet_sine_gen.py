from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

rng = np.random.default_rng(2)

# -------------------------------------- settings -------------------------------------
RESOLUTIONS = [32, 64, 128, 256]

SAMPLES = 128
SENSOR_SAMPLES = 32
NOISE = 0.2

sensor_locs = np.linspace(-1, 1, SENSOR_SAMPLES)
shifts = rng.uniform(0, 1, SAMPLES)

f = lambda x, shift: np.sin(2 * np.pi * (x + shift))
df = lambda x, shift: 2 * np.pi * np.cos(2 * np.pi * (x + shift))

# ------------------------------------ create data ------------------------------------
# sensor values are shared across resolutions (sensors are fixed by architecture)
G = np.zeros((SAMPLES, SENSOR_SAMPLES))
for i, shift in enumerate(shifts):
    G[i] = f(sensor_locs, shift) + NOISE * rng.uniform(-1, 1, SENSOR_SAMPLES)

for resolution in RESOLUTIONS:
    X = np.zeros((SAMPLES, resolution))
    Y = np.zeros((SAMPLES, resolution))
    for i, shift in enumerate(shifts):
        X[i] = rng.uniform(-1, 1, resolution)
        Y[i] = df(X[i], shift) + NOISE * rng.uniform(-1, 1, resolution)

    np.savez(DATA_DIR / f"deeponet_sine_{resolution}.npz", X=X, Y=Y, G=G)
