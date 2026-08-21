from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
RESOLUTIONS = [32, 64, 128]
LENGTH = 1.0
RADIUS = 0.2

# -------------------------------------- create data ----------------------------------
for resolution in RESOLUTIONS:
    h = LENGTH / resolution
    xc = np.linspace(h / 2, LENGTH - h / 2, resolution)
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    r2 = (X - LENGTH / 2) ** 2 + (Y - LENGTH / 2) ** 2 + (Z - LENGTH / 2) ** 2
    indicator = np.where(r2 < RADIUS**2, 0, 255).astype(np.uint8)

# --------------------------------------- export --------------------------------------
    out = DATA_DIR / f"void_cube_{resolution}.npz"
    np.savez(out, indicator=indicator, Lx=LENGTH, Ly=LENGTH, Lz=LENGTH)
    solid_fraction = (indicator > 0).mean()
    exact = 1.0 - 4.0 / 3.0 * np.pi * RADIUS**3 / LENGTH**3
    print(f"\tsaved {out}\n\tsolid fraction {solid_fraction:.5f} (exact {exact:.5f})")
