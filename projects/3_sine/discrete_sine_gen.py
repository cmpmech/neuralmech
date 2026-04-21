from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"

np.random.seed(1)

f = lambda x: np.sin(2 * np.pi * x)


def discretize(y, classes):
    thresholds = np.linspace(-1, 1, classes + 1)
    y_ = np.zeros_like(y)
    for threshold in thresholds[1:-1]:
        y_[y > threshold] += 1
    return y_.astype(np.long)


SAMPLES = 128
CLASSES = 3
NOISE = 0.0

# ---------------------- training & validation data ----------------------
X = np.random.uniform(-1, 1, (SAMPLES, 1))
Y = discretize(f(X) + NOISE * np.random.uniform(-1, 1, (SAMPLES, 1)), CLASSES)
np.savez(DATA_DIR / f"discrete_sine.npz", X=X, Y=Y.squeeze())

# ----------------------------- testing data -----------------------------
X = np.expand_dims(np.linspace(-1.3, 1.3, 1000), 1)
Y = discretize(f(X) + NOISE * np.random.uniform(-1, 1, (1000, 1)), CLASSES)
np.savez(DATA_DIR / f"discrete_sine_test.npz", X=X, Y=Y.squeeze())
