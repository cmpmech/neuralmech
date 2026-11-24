import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
EPOCH = "https://epoch.ai/data/benchmark_data.zip"
BENCHMARKS = {  # name: (file, score column)
    "mmlu": ("mmlu_external.csv", "EM"),
    "gpqa": ("gpqa_diamond.csv", "Best score (across scorers)"),
    "frontiermath": ("frontiermath_tiers_1_3_v2.csv", "Best score (across scorers)"),
    "swebench": ("swe_bench_verified.csv", "Best score (across scorers)"),
    "arcagi2": ("arc_agi_2_external.csv", "Score"),
    "hle": ("hle_external.csv", "Accuracy"),
}

# -------------------------------------- load data ------------------------------------
request = urllib.request.Request(EPOCH, headers={"User-Agent": "neuralmech"})
with urllib.request.urlopen(request) as response:
    archive = zipfile.ZipFile(io.BytesIO(response.read()))

# ----------------------------------- postprocessing ----------------------------------
frontiers = {}
for name, (file, column) in BENCHMARKS.items():
    df = pd.read_csv(archive.open(file))
    day = pd.to_datetime(df["Release date"], errors="coerce")
    df["year"] = day.dt.year + (day.dt.dayofyear - 1) / 365.25
    df = df.dropna(subset=["year", column]).sort_values("year")
    best = df[column].cummax() * 100
    keep = best > best.shift(fill_value=-1)
    frontier = pd.DataFrame({"x": df.year[keep], "y": best[keep]}).reset_index(drop=True)
    frontier.loc[len(frontier)] = [df.year.max(), best.iloc[-1]]
    frontiers[name] = frontier
    print(f"{name}: {frontier.y.iloc[0]:.0f} % in {frontier.x.iloc[0]:.1f}, "
          f"{frontier.y.iloc[-1]:.0f} % in {frontier.x.iloc[-1]:.1f}")

if not args.book:
    for frontier in frontiers.values():
        plt.step(frontier.x, frontier.y, where="post")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, frontier in frontiers.items():
        x, y = frontier.x.to_numpy(), frontier.y.to_numpy()
        save_csv(CSV_DIR / f"benchmark_{name}.csv", x=x, y=y)
