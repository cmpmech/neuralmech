from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"

# ------------------------------------ ct settings ------------------------------------
# Nx = 80
# Ny = 40
# Nz = 20

Nx = 160
Ny = 80
Nz = 40

Lx = 2.0
Ly = 1.0
Lz = 0.5
min_indicator = 1e-5

# ------------------------------------- create ct -------------------------------------
sx, sy, sz = Lx / Nx, Ly / Ny, Lz / Nz
xc = np.linspace(sx / 2, Lx - sx / 2, Nx)
yc = np.linspace(sy / 2, Ly - sy / 2, Ny)
zc = np.linspace(sz / 2, Lz - sz / 2, Nz)
X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
indicator = np.where(X**2 + (Y - 1.0) ** 2 + Z**2 < 0.25, min_indicator, 1.0)

# --------------------------------------- export --------------------------------------
out = DATA_DIR / "plate_hole_3D.npz"
np.savez(out, indicator=indicator, Lx=Lx, Ly=Ly, Lz=Lz)
print(f"\tsaved {out}\n\tshape={indicator.shape}\n\tvoxels={np.prod(indicator.shape)}")
