from pathlib import Path

import numpy as np
from pybliometrics import init
from pybliometrics.scopus import ScopusSearch

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

init()

# -------------------------------------- settings -------------------------------------
TERMS = {
    "artificial_intelligence": '"artificial intelligence"',
    "machine_learning": '"machine learning"',
    "deep_learning": '"deep learning"',
    "neural_network": '"neural network"',
    "physics-informed_neural_network": '"physics-informed neural network"',
    "artificial_general_intelligence": '"artificial general intelligence"',
}
YEARS = np.arange(1940, 2026)

# ------------------------------------ create data ------------------------------------
results = {}
for label, term in TERMS.items():
    print(f"fetching {label}")
    counts = []
    for year in YEARS:
        query = f"TITLE-ABS-KEY({term}) AND PUBYEAR = {year}"
        s = ScopusSearch(query, download=False)
        counts.append(s.get_results_size())
    results[label] = counts

# --------------------------------------- export --------------------------------------
save_csv(CSV_DIR / "ai_in_science.csv", year=YEARS, **results)
