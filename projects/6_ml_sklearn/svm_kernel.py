import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
SAMPLES = 150
INNER = (0.0, 1.0)  # radius range of class 1
OUTER = (1.5, 2.2)  # radius range of class -1
CENTER = (0.15, 0.0)  # off the origin, so separating plane tilts
LIMIT = 2.4
GAMMA = 0.5  # rbf kernel exp(-GAMMA |x - x'|^2), length scale 1
RESOLUTION = 200

# ------------------------------------ create data ------------------------------------
y = np.where(np.arange(SAMPLES) < SAMPLES // 3, 1, -1)
low = np.where(y == 1, INNER[0], OUTER[0])
high = np.where(y == 1, INNER[1], OUTER[1])
radius = np.sqrt(rng.uniform(low**2, high**2))  # uniform density in each ring
angle = rng.uniform(0, 2 * np.pi, SAMPLES)
X = CENTER + radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)

# -------------------------------------- training -------------------------------------
lifted = np.column_stack([X, (X**2).sum(1)])
model_lifted = SVC(kernel="linear", C=1e3).fit(lifted, y)
model_rbf = SVC(kernel="rbf", gamma=GAMMA, C=1e3).fit(X, y)
print(f"lifted support vectors {len(model_lifted.support_)}")
print(f"rbf support vectors {len(model_rbf.support_)}")

# ----------------------------------- postprocessing ----------------------------------
w = model_lifted.coef_[0]
b = model_lifted.intercept_[0]
plane = -np.array([b, *w[:2]]) / w[2]
print(f"lifted plane z = {plane[0]:.3f} + {plane[1]:.4f} x1 + {plane[2]:.4f} x2")
t = np.linspace(0, 2 * np.pi, 200)
center = -w[:2] / (2 * w[2])
circle_radius = np.sqrt(center @ center - b / w[2])
circle = center + circle_radius * np.stack([np.cos(t), np.sin(t)], axis=1)

x_test = np.linspace(-LIMIT, LIMIT, RESOLUTION)
x1, x2 = np.meshgrid(x_test, x_test, indexing="ij")
y_pred_test = model_rbf.decision_function(np.column_stack([x1.ravel(), x2.ravel()]))
y_pred_test = y_pred_test.reshape(x1.shape)

if not args.book:
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(131)
    ax.scatter(*X.T, c=y, cmap="coolwarm")
    ax = fig.add_subplot(132, projection="3d")
    ax.scatter(*lifted.T, c=y, cmap="coolwarm")
    ax.plot(*circle.T, 0, "k")
    ax.plot_surface(x1, x2, -(w[0] * x1 + w[1] * x2 + b) / w[2], alpha=0.3)
    ax = fig.add_subplot(133)
    ax.contour(x1, x2, y_pred_test, levels=[-1, 0, 1], colors="k")
    ax.scatter(*X.T, c=y, cmap="coolwarm")
    ax.scatter(*model_rbf.support_vectors_.T, facecolors="none", edgecolors="k", s=80)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    contours = plt.contour(x1, x2, y_pred_test, levels=[-1, 0, 1])
    for level, segments in zip(["minus", "zero", "plus"], contours.allsegs):
        line = np.concatenate([np.vstack([s, [np.nan, np.nan]]) for s in segments])
        path = CSV_DIR / f"svm_kernel_level_{level}.csv"
        save_csv(path, x1=line[:, 0], x2=line[:, 1])
    save_csv(CSV_DIR / "svm_kernel_circle.csv", x1=circle[:, 0], x2=circle[:, 1])
    support = np.zeros(SAMPLES, bool)
    support[model_rbf.support_] = True
    sets = [("in", y == 1), ("out", y == -1), ("support", support)]
    for name, ids in sets:
        path = CSV_DIR / f"svm_kernel_{name}.csv"
        save_csv(path, x1=X[ids, 0], x2=X[ids, 1], z=lifted[ids, 2])
