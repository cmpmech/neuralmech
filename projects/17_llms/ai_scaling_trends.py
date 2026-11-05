import argparse
import urllib.request
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
OWID = "https://ourworldindata.org/grapher/{}.csv?v=1&csvType=full&useColumnShortNames=false"
SERIES = {
    "compute": ("exponential-growth-of-computation-in-the-training-of-notable-ai-systems",
                "Training computation (petaFLOP)"),
    "data": ("exponential-growth-of-datapoints-used-to-train-notable-ai-systems",
             "Training dataset size"),
    "params": ("exponential-growth-of-parameters-in-notable-ai-systems", "Number of parameters"),
}
EPOCH = "https://epoch.ai/data/notable_ai_models.csv"
COST = "Training compute cost (2023 USD)"
BREAK = 2010  # separate trends before and after deep learning
COST_START = 2016


# --------------------------------------- helper --------------------------------------
def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "neuralmech"})
    with urllib.request.urlopen(request) as response:
        return pd.read_csv(StringIO(response.read().decode()))


def fractional_year(day):
    day = pd.to_datetime(day)
    return (day.dt.year + (day.dt.dayofyear - 1) / 365.25).to_numpy()


def fit(x, y, grid):
    slope, intercept = np.polyfit(x, np.log10(y), 1)
    return slope, 10 ** (slope * grid + intercept)


# -------------------------------------- load data ------------------------------------
points = {}
for key, (name, column) in SERIES.items():
    df = download(OWID.format(name))
    df = df[~df.Entity.str.contains("x/year")]  # trend rows of the chart
    x, y = fractional_year(df.Day), df[column].to_numpy()
    points[key] = x[y > 0], y[y > 0]

df = download(EPOCH)
df = df[(df["Frontier model"] == True) & df[COST].notna()]
x, y = fractional_year(df["Publication date"]), df[COST].to_numpy()
points["money"] = x[x >= COST_START], y[x >= COST_START]

# ----------------------------------- postprocessing ----------------------------------
fits = {}
for key in SERIES:
    x, y = points[key]
    x1, x2 = np.array([1950.0, BREAK]), np.array([BREAK, 2026.0])
    w1, y1 = fit(x[x < BREAK], y[x < BREAK], x1)
    w2, y2 = fit(x[x > BREAK], y[x > BREAK], x2)
    fits[key] = dict(x1=x1, y1=y1, x2=x2, y2=y2)
    print(f"{key}: {len(x)} models, doubling {12 * np.log10(2) / w1:.1f} months before "
          f"{BREAK}, {12 * np.log10(2) / w2:.1f} months since")
x, y = points["money"]
grid = np.array([COST_START, 2026.0])
w, y_fit = fit(x, y, grid)
fits["money"] = dict(x=grid, y=y_fit)
print(f"money: {len(x)} models, growth {10**w:.2f}x per year")

if not args.book:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key in zip(axes.flat, ["compute", "data", "params", "money"]):
        ax.semilogy(*points[key], "o", mfc="none", alpha=0.5)
        if key == "money":
            ax.semilogy(fits[key]["x"], fits[key]["y"], "k")
        else:
            ax.semilogy(fits[key]["x1"], fits[key]["y1"], "b")
            ax.semilogy(fits[key]["x2"], fits[key]["y2"], "k")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for key in points:
        save_csv(CSV_DIR / f"scaling_{key}.csv", x=points[key][0], y=points[key][1])
        save_csv(CSV_DIR / f"scalingfits_{key}.csv", **fits[key])
