import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 50000
LR = 0.1
REGULARIZATION = 1e-4

SAMPLES = 200
LIMIT = 1.6  # stresses drawn from [-LIMIT, LIMIT]^2, in units of the yield stress


# --------------------------------------- helper --------------------------------------
def train_svm(features, y):
    w = np.zeros(features.shape[1])
    b = 0.0
    for epoch in range(EPOCHS):
        active = y * (features @ w + b) < 1  # inside margin or misclassified
        dw = REGULARIZATION * w - (y[active, None] * features[active]).sum(0) / len(y)
        db = -y[active].sum() / len(y)
        w -= LR * dw
        b -= LR * db
    return w, b


def lift(s):
    return np.stack([s[:, 0] ** 2, s[:, 1] ** 2, s[:, 0] * s[:, 1]], axis=1)


def von_mises(s):
    return s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2


# ------------------------------------ create data ------------------------------------
X = rng.uniform(-LIMIT, LIMIT, (SAMPLES, 2))  # principal stresses sigma_1, sigma_2
y = np.where(von_mises(X) <= 1, 1, -1)  # elastic +1, plastic -1

# -------------------------------------- training -------------------------------------
w_linear, b_linear = train_svm(X, y)
w, b = train_svm(lift(X), y)

f = lift(X) @ w + b
support = np.abs(y * f - 1) < 0.05  # samples on the margin
accuracy_linear = np.mean(np.sign(X @ w_linear + b_linear) == y)
accuracy = np.mean(np.sign(f) == y)
print(f"linear accuracy {accuracy_linear:.2f}")
print(f"quadratic accuracy {accuracy:.2f}, {support.sum()} support vectors")
a = -w / b
print(f"quadratic boundary {a[0]:.3f} s1^2 + {a[1]:.3f} s2^2 + {a[2]:.3f} s1 s2 = 1")

# ----------------------------------- postprocessing ----------------------------------
angle = np.linspace(0, 2 * np.pi, 200)
direction = np.stack([np.cos(angle), np.sin(angle)], axis=1)
ellipse_svm = direction / np.sqrt(-(lift(direction) @ w) / b)[:, None]
ellipse_true = direction / np.sqrt(von_mises(direction))[:, None]

s1 = np.array([-LIMIT, LIMIT])
line = np.stack([s1, -(w_linear[0] * s1 + b_linear) / w_linear[1]], axis=1)
z = -(lift(X) @ w) / b  # lifted third coordinate, the plane z = 1 separates

if not args.book:
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(131)
    ax.scatter(*X.T, c=y, cmap="coolwarm")
    ax.plot(*line.T, "k")
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax = fig.add_subplot(132, projection="3d")
    ax.scatter(*X.T, z, c=y, cmap="coolwarm")
    ax = fig.add_subplot(133)
    ax.scatter(*X.T, c=y, cmap="coolwarm")
    ax.scatter(*X[support].T, facecolors="none", edgecolors="k", s=80)
    ax.plot(*ellipse_svm.T, "k")
    ax.plot(*ellipse_true.T, "r--")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    sets = [("elastic", y == 1), ("plastic", y == -1), ("support", support)]
    for name, ids in sets:
        save_csv(CSV_DIR / f"svm_{name}.csv", s1=X[ids, 0], s2=X[ids, 1], z=z[ids])
    save_csv(CSV_DIR / "svm_linear.csv", s1=line[:, 0], s2=line[:, 1])
    save_csv(CSV_DIR / "svm_quadratic.csv", s1=ellipse_svm[:, 0], s2=ellipse_svm[:, 1])
    save_csv(CSV_DIR / "svm_vonmises.csv", s1=ellipse_true[:, 0], s2=ellipse_true[:, 1])
