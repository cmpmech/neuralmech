import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import KNeighborsRegressor

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "../10_rom"))  # cantilever beam finite elements

from helper import beam_matrices

RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
ELEMENTS = 50
MODES = 3
SAMPLES = 300
TEST_SAMPLES = 300
SEVERITY = (0.1, 0.5)  # stiffness reduction of the damaged element
NOISE = 2e-4  # relative measurement noise of the eigenfrequencies
NEIGHBORS = 5
QUERY = 12  # test sample whose neighbors are shown

rng = np.random.default_rng(0)


# -------------------------------------- helper ---------------------------------------
def frequencies(EI):
    K, M = beam_matrices(EI)
    eigenvalues = eigh(K.toarray(), M.toarray(), eigvals_only=True, subset_by_index=[0, MODES - 1])
    return np.sqrt(eigenvalues)


def modes():
    K, M = beam_matrices(np.ones(ELEMENTS))
    _, vectors = eigh(K.toarray(), M.toarray(), subset_by_index=[0, MODES - 1])
    deflection = np.vstack([np.zeros(MODES), vectors[0::2]])
    return deflection / deflection[-1]


def create_data(samples):
    element = rng.integers(0, ELEMENTS, samples)
    severity = rng.uniform(*SEVERITY, samples)
    shifts = np.zeros((samples, MODES))
    for i in range(samples):
        EI = np.ones(ELEMENTS)
        EI[element[i]] = 1 - severity[i]
        measured = frequencies(EI) * (1 + NOISE * rng.standard_normal(MODES))
        shifts[i] = 1 - measured / intact
    features = shifts / np.linalg.norm(shifts, axis=1, keepdims=True)  # independent of severity
    location = (element + 0.5) / ELEMENTS
    return features, location


# ------------------------------------ create data ------------------------------------
intact = frequencies(np.ones(ELEMENTS))
X, location = create_data(SAMPLES)
X_test, location_test = create_data(TEST_SAMPLES)

# ------------------------------------- training --------------------------------------
knn = KNeighborsRegressor(n_neighbors=NEIGHBORS).fit(X, location)
location_pred = knn.predict(X_test)
error = np.abs(location_pred - location_test)
tip = location_test > 0.8
print(f"mean location error {error[~tip].mean():.3f} (x <= 0.8), {error[tip].mean():.3f} (x > 0.8)")

neighbors = knn.kneighbors(X_test[QUERY:QUERY + 1], return_distance=False)[0]
print(f"query at x = {location_test[QUERY]:.2f}, predicted {location_pred[QUERY]:.2f}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(X[:, 0], X[:, 1], c=location, s=4, cmap="cividis")
    axes[0].scatter(*X[neighbors, :2].T, facecolors="none", edgecolors="r", s=60)
    axes[0].scatter(*X_test[QUERY, :2], c="r", marker="x")
    axes[1].scatter(location_test, location_pred, s=4)
    axes[1].plot([0, 1], [0, 1], "k--")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "knn_damage_train.csv", f1=X[:, 0], f2=X[:, 1], x=location)
    save_csv(CSV_DIR / "knn_damage_neighbors.csv", f1=X[neighbors, 0], f2=X[neighbors, 1])
    save_csv(CSV_DIR / "knn_damage_query.csv", f1=X_test[QUERY:QUERY + 1, 0], f2=X_test[QUERY:QUERY + 1, 1])
    save_csv(CSV_DIR / "knn_damage_test.csv", x=location_test, x_pred=location_pred)
    deflection = modes()
    nodes = np.linspace(0, 1, ELEMENTS + 1)
    save_csv(CSV_DIR / "knn_damage_modes.csv", x=nodes, **{f"w{i + 1}": deflection[:, i] for i in range(MODES)})
