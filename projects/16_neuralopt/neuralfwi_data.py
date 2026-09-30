import time
from pathlib import Path

import cupy as cp
import numpy as np
from tqdm import tqdm
from cuwave.utils import misfit_gradient

from helper import FWI

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
SAMPLES = 2000
SEED = 1  # the evaluation case in neuralfwi.py is the cuwave circle stack, not a sample

# ------------------------------------ create data ------------------------------------
fwi = FWI()
rng = np.random.default_rng(SEED)
gamma = cp.ones(fwi.sim.Nx_padded, dtype=fwi.sim.dtype)
gradients = np.zeros((SAMPLES, *fwi.sim.Nx_padded), dtype=np.float16)
truths = np.zeros((SAMPLES, *fwi.sim.Nx_padded), dtype=bool)

tic = time.time()
for i in tqdm(range(SAMPLES)):
    truth = fwi.ellipses(rng)
    observed = fwi.observe(truth)
    _, gradient = misfit_gradient(fwi.sim, fwi.sources, gamma, fwi.sensors, observed)
    gradients[i] = (gradient / cp.abs(gradient).max()).get()
    truths[i] = (truth < 0.5).get()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# --------------------------------------- export --------------------------------------
np.savez(DATA_DIR / "neuralfwi.npz", gradients=gradients, voids=truths)
print(f"saved to {DATA_DIR / 'neuralfwi.npz'}")
