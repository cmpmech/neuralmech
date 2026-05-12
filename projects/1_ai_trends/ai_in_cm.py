from pathlib import Path

import numpy as np
from pybliometrics import init
from pybliometrics.scopus import ScopusSearch

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

init()

terms = {
    "artificial_intelligence": '"artificial intelligence"',
    "machine_learning": '"machine learning"',
    "deep_learning": '"deep learning"',
    "neural_network": '"neural network"',
    "physics-informed_neural_network": '"physics-informed neural network"',
    "artificial-general_intelligence": '"artificial general intelligence"',
}
years = np.arange(1940, 2026)
cm_journals = [
    "Computer Methods in Applied Mechanics and Engineering",
    "International Journal for Numerical Methods in Engineering",
    "Computational Mechanics",
    "Finite Elements in Analysis and Design",
    "International Journal of Numerical Methods in Fluids",
    "Journal of Computational Physics",
    "Engineering with Computers",
    "Computers and Structures",
]

journal_filter = " OR ".join([f'SRCTITLE("{j}")' for j in cm_journals])
results = {}
for label, term in terms.items():
    print(f"Fetching: {label}")
    counts = []
    for year in years:
        query = f"TITLE-ABS-KEY({term}) AND PUBYEAR = {year} AND ({journal_filter})"
        s = ScopusSearch(query, download=False)
        counts.append(s.get_results_size())
    results[label] = counts

save_csv(RESULTS_DIR / "ai_in_cm.csv", year=years, **results)
