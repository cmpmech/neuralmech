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

# -------------------------------------- settings -------------------------------------
N = 12  # number of key-value pairs
TEMPERATURES = [0.5, 0.05, 0.005]
QUERY = 0.4  # position of the query whose weights are shown
RESOLUTION = 400

# ------------------------------------ create data ------------------------------------
rng = np.random.default_rng(2)
x_keys = np.sort(rng.uniform(0, 1, N))
values = np.sin(2 * np.pi * x_keys)
x_queries = np.linspace(0, 1, RESOLUTION)


def embed(x):
    # unit vectors on a half circle, so that the dot product decays with distance in x
    return np.stack([np.cos(np.pi * x), np.sin(np.pi * x)], axis=-1)


def attention(x_query, temperature):
    scores = embed(x_query) @ embed(x_keys).T / temperature
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights, weights @ values


# ----------------------------------- postprocessing ----------------------------------
weights = [attention(np.array([QUERY]), T)[0][0] for T in TEMPERATURES]
outputs = [attention(x_queries, T)[1] for T in TEMPERATURES]

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for w, y in zip(weights, outputs):
        axes[0].plot(x_keys, w, "o-")
        axes[1].plot(x_queries, y)
    axes[0].axvline(QUERY, color="k", ls="--")
    axes[1].plot(x_keys, values, "ko")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "attention_kernel_values.csv", x=x_keys, v=values)
    save_csv(
        CSV_DIR / "attention_kernel_weights.csv",
        x=x_keys,
        **{f"a{k}": w for k, w in enumerate(weights)},
    )
    save_csv(
        CSV_DIR / "attention_kernel_outputs.csv",
        x=x_queries,
        **{f"y{k}": y for k, y in enumerate(outputs)},
    )
