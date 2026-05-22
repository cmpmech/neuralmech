from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"

# ------------------------------------ ct settings ------------------------------------
Nx = 160
Ny = 80
Lx = 2.0
Ly = 1.0

# ------------------------------------- create ct -------------------------------------
sx, sy = Lx / Nx, Ly / Ny
xc = np.linspace(sx / 2, Lx - sx / 2, Nx)
yc = np.linspace(sy / 2, Ly - sy / 2, Ny)
X, Y = np.meshgrid(xc, yc, indexing="ij")
indicator = np.where(X**2 + (Y - 1.0) ** 2 < 0.25, 0, 255).astype(np.uint8)

# --------------------------------------- export --------------------------------------
out = DATA_DIR / "plate_hole_2D.npz"
np.savez(out, indicator=indicator, Lx=Lx, Ly=Ly)
print(f"\tsaved {out}\n\tshape={indicator.shape}\n\tvoxels={np.prod(indicator.shape)}")

# ----------------------------------- postprocessing ----------------------------------
# fig, ax = plt.subplots()
# ax.imshow(indicator.T, origin="lower", cmap="binary", vmin=0, vmax=255)
# ax.axis("off")
# fig.tight_layout(pad=0)
# plt.show()
