from pathlib import Path

import nrrd
import numpy as np
from scipy.ndimage import label

BASE_DIR = Path(__file__).parent
EXT_DATA_DIR = BASE_DIR / "../../external_data"
DATA_DIR = BASE_DIR / "../../data"

# ------------------------------------ ct settings ------------------------------------
CT_FILE = "B-HAI-1.nrrd"
CROP = (slice(None), slice(256, 772), slice(1, 1060))  # (z_all, row_crop, col_crop)
SEED = (5, 5, 5)   # must be solid
SOLID_STRIP = 3    # voxels of guaranteed solid at x=Lx

# --------------------------------------- load ----------------------------------------
data, _ = nrrd.read(EXT_DATA_DIR / CT_FILE)
data = np.array(data)

# ----------------------------------- binarize ----------------------------------------
data = (data - data.min()) / (data.max() - data.min())
data = (data > 0.5).astype(np.uint8)

# --------------------------------------- crop ----------------------------------------
data = data[CROP]           # (Nz, Nx_crop, Ny_crop)
data = data.transpose(1, 2, 0)  # → (Nx, Ny, Nz)

# ----------------------------------- bfs cleaning ------------------------------------
labeled, _ = label(data)
seed_label = labeled[SEED]
data[labeled != seed_label] = 0

# --------------------------------- orient + trim -------------------------------------
rows = np.where(np.any(data, axis=(1, 2)))[0]
cols = np.where(np.any(data, axis=(0, 2)))[0]
deps = np.where(np.any(data, axis=(0, 1)))[0]
data = data[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1, deps[0] : deps[-1] + 1]

# -------------------------- solid strip at right face --------------------------------
data[-SOLID_STRIP:, :, :] = 1

# --------------------------------------- export --------------------------------------
Nx, Ny, Nz = data.shape
Lx = 1.0
Ly = Ny / Nx
Lz = Nz / Nx
indicator = (data * 255).astype(np.uint8)

out = DATA_DIR / f"{Path(CT_FILE).stem.replace('-', '_')}_3D.npz"
np.savez(out, indicator=indicator, Lx=Lx, Ly=Ly, Lz=Lz)
print(f"\tsaved {out}\n\tshape={indicator.shape}\n\tvoxels={np.prod(indicator.shape)}")
