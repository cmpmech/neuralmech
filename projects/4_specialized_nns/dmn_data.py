from pathlib import Path

import numpy as np
from tqdm import tqdm

from solvers.homogenization import effective_stiffness

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
SAMPLES = 300

# microstructure: centered circular phase-2 inclusion in a phase-1 matrix
RADIUS = 0.25
NELEMENTS = 12
DEGREE = 2

# phase sampling: log-uniform moduli with a soft-but-not-void inclusion (moderate
# contrast keeps the homogenization well conditioned), moderate Poisson ratios
E1_RANGE = (0.8, 1.25)
E2_RANGE = (0.1, 1.0)
NU_RANGE = (0.15, 0.35)

# ------------------------------------ helper -----------------------------------------
def isotropic_stiffness(E, nu):
    return E / (1 - nu**2) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
    )


# ----------------------------------- create data -------------------------------------
E1 = 10 ** rng.uniform(np.log10(E1_RANGE[0]), np.log10(E1_RANGE[1]), SAMPLES)
E2 = 10 ** rng.uniform(np.log10(E2_RANGE[0]), np.log10(E2_RANGE[1]), SAMPLES)
nu1 = rng.uniform(*NU_RANGE, SAMPLES)
nu2 = rng.uniform(*NU_RANGE, SAMPLES)

C1 = np.zeros((SAMPLES, 3, 3))
C2 = np.zeros((SAMPLES, 3, 3))
C_eff = np.zeros((SAMPLES, 3, 3))

for i in tqdm(range(SAMPLES)):
    C1[i] = isotropic_stiffness(E1[i], nu1[i])
    C2[i] = isotropic_stiffness(E2[i], nu2[i])
    C_eff[i] = effective_stiffness(
        E1[i], nu1[i], E2[i], nu2[i], radius=RADIUS, nelements=NELEMENTS, degree=DEGREE
    )

# ------------------------------------- export ----------------------------------------
np.savez(DATA_DIR / "dmn_dataset.npz", C1=C1, C2=C2, C_eff=C_eff)
print(f"saved {SAMPLES} samples to {DATA_DIR / 'dmn_dataset.npz'}")
