from pathlib import Path

import numpy as np
from pybliometrics import init
from pybliometrics.scopus import ScopusSearch

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

init()

# -------------------------------------- settings -------------------------------------
TERMS = {
    "artificial_intelligence": '"artificial intelligence"',
    "machine_learning": '"machine learning"',
    "deep_learning": '"deep learning"',
    "neural_network": '"neural network"',
    "physics-informed_neural_network": '"physics-informed neural network"',
    "artificial-general_intelligence": '"artificial general intelligence"',
}
YEARS = np.arange(1940, 2026)
CM_JOURNALS = [
    "Computer Methods in Applied Mechanics and Engineering",
    "International Journal for Numerical Methods in Engineering",
    "Computational Mechanics",
    "Finite Elements in Analysis and Design",
    "International Journal of Numerical Methods in Fluids",
    "Journal of Computational Physics",
    "Engineering with Computers",
    "Computers and Structures",
]

# ------------------------------------ create data ------------------------------------
# restrict the search to representative computational mechanics journals
journal_filter = " OR ".join([f'SRCTITLE("{j}")' for j in CM_JOURNALS])
results = {}
for label, term in TERMS.items():
    print(f"fetching {label}")
    counts = []
    for year in YEARS:
        query = f"TITLE-ABS-KEY({term}) AND PUBYEAR = {year} AND ({journal_filter})"
        s = ScopusSearch(query, download=False)
        counts.append(s.get_results_size())
    results[label] = counts

# --------------------------------------- export --------------------------------------
save_csv(RESULTS_DIR / "ai_in_cm.csv", year=YEARS, **results)
