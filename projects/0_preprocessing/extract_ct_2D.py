from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
from scipy.ndimage import label

BASE_DIR = Path(__file__).parent
EXT_DATA_DIR = BASE_DIR / "../../external_data"
DATA_DIR = BASE_DIR / "../../data"

# ------------------------------------ ct settings ------------------------------------
CT_FILE = "B-HAI-1.nrrd"
CROP = (slice(256, 772), slice(1, 1060))
SEED = (10, 10)  # must be solid
SOLID_STRIP = 3  # voxels of guaranteed solid at x=0 and x=Lx

# ---------------------------------- load + slice -------------------------------------
data, _ = nrrd.read(EXT_DATA_DIR / CT_FILE)
data = np.array(data)[data.shape[0] // 2, :, :]

# ----------------------------------- binarize ----------------------------------------
data = (data - data.min()) / (data.max() - data.min())
data = (data > 0.5).astype(np.uint8)

# --------------------------------------- crop ----------------------------------------
data = data[CROP]

# ----------------------------------- bfs cleaning ------------------------------------
labeled, _ = label(data)
seed_label = labeled[SEED]
data[labeled != seed_label] = 0

# --------------------------------- orient + trim -------------------------------------
data = data.T  # swap x/y so longer axis becomes x

# trim void border rows/cols so x=0 and y=0 faces touch solid
rows = np.where(np.any(data, axis=1))[0]
cols = np.where(np.any(data, axis=0))[0]
data = data[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

# -------------------------- solid strip at left and right face -----------------------
# data[:SOLID_STRIP, :] = 1 # not needed
data[-SOLID_STRIP:, :] = 1

# --------------------------------------- export --------------------------------------
Nx, Ny = data.shape
Lx = 1.0
Ly = Ny / Nx
indicator = (data * 255).astype(np.uint8)

out = DATA_DIR / f"{Path(CT_FILE).stem.replace('-', '_')}_2D.npz"
np.savez(out, indicator=indicator, Lx=Lx, Ly=Ly)
print(f"\tsaved {out}\n\tshape={indicator.shape}\n\tvoxels={np.prod(indicator.shape)}")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.imshow(indicator.T, origin="lower", cmap="binary", vmin=0, vmax=255)
ax.axis("off")
fig.tight_layout(pad=0)
plt.show()
