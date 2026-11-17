import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"

# ------------------------------------- load data -------------------------------------
archive = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=60).content))
table = pd.read_excel(archive.open("Concrete_Data.xls")).to_numpy()

# --------------------------------------- export --------------------------------------
np.savez(DATA_DIR / "concrete.npz", X=table[:, :8], Y=table[:, 8])
print(f"saved {DATA_DIR / 'concrete.npz'}")
