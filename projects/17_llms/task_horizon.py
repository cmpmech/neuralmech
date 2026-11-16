import argparse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
METR = "https://metr.org/assets/benchmark_results_1_1.yaml"
LIMIT = 16 * 60  # minutes, above which the task suite no longer measures reliably
BREAK = 2023.0


# --------------------------------------- helper --------------------------------------
def fit(x, y, grid):
    slope, intercept = np.polyfit(x, np.log2(y), 1)
    return 365.25 / slope, 2 ** (slope * grid + intercept)


# -------------------------------------- load data ------------------------------------
request = urllib.request.Request(METR, headers={"User-Agent": "neuralmech"})
with urllib.request.urlopen(request) as response:
    results = yaml.safe_load(response.read())["results"]
rows = [dict(date=r["release_date"], minutes=r["metrics"]["p50_horizon_length"]["estimate"],
             sota=r["metrics"].get("is_sota", False)) for r in results.values()]
df = pd.DataFrame(rows)
day = pd.to_datetime(df.date)
df["year"] = day.dt.year + (day.dt.dayofyear - 1) / 365.25

# ----------------------------------- postprocessing ----------------------------------
frontier = df[df.sota & (df.minutes < LIMIT)]
x1, x2 = np.array([frontier.year.min(), 2026.5]), np.array([BREAK, 2026.5])
days1, y1 = fit(frontier.year, frontier.minutes, x1)
late = frontier[frontier.year >= BREAK]
days2, y2 = fit(late.year, late.minutes, x2)
print(f"{len(df)} models, {len(frontier)} frontier; doubling every {days1:.0f} days since "
      f"{frontier.year.min():.0f}, every {days2:.0f} days since {BREAK:.0f}")
print(df.sort_values("year").tail(5).to_string(index=False))

if not args.book:
    plt.semilogy(df.year, df.minutes, "o", mfc="none")
    plt.semilogy(x1, y1, "b")
    plt.semilogy(x2, y2, "k")
    plt.axhline(LIMIT, color="k", ls=":")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "task_horizon.csv", x=df.year.to_numpy(), y=df.minutes.to_numpy())
    save_csv(CSV_DIR / "task_horizon_fit.csv", x1=x1, y1=y1, x2=x2, y2=y2)
