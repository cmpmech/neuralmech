from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
EXTERNAL_DATA_DIR = (BASE_DIR / "../../external_data").resolve()

# ------------------------------------- load data -------------------------------------
# uci concrete compressive strength: 8 mixture components -> strength in mpa
data = pd.read_excel(EXTERNAL_DATA_DIR / "Concrete_Data.xls").values

# ------------------------------------ create data ------------------------------------
path = DATA_DIR / "concrete.npz"
np.savez(path, X=data[:, :-1], Y=data[:, -1])
print(f"saved {path}")
