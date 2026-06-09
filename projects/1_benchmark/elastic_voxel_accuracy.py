import argparse
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
STL_DISP_DIR = DATA_DIR / "elasticity/solution_stl/displacement"
VOXEL_DISP_DIR = DATA_DIR / "elasticity/solution_voxel/displacement"

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--case", type=int, default=1)
args = parser.parse_args()

name = f"{args.geometry:09}_abc"

# ----------------------------------- relative L2 error -------------------------------
# Both solvers write displacement on the SAME (N+1)^3 nodal voxel grid; the STL solve is
# the FCM ground truth, the voxel solve the staircased benchmark. Void nodes are zero in
# both fields, so a plain nodal mean is a node-lumped quadrature of the L2 integral and
# the (equal) cell volume cancels in the ratio -- no geometry/spacing needed.
ref = np.load(STL_DISP_DIR / f"{name}_{args.case}.npz")["displacement"].astype(
    np.float64
)
vox = np.load(VOXEL_DISP_DIR / f"{name}_{args.case}.npz")["displacement"].astype(
    np.float64
)

ORD = 2
# ORD = "inf"

mag_err = np.linalg.norm(ref - vox, axis=-1)  # per-node error magnitude
mag_ref = np.linalg.norm(ref, axis=-1)
if ORD == 2:
    rel = np.sqrt((mag_err**2).mean()) / np.sqrt((mag_ref**2).mean())
elif ORD == "inf":
    rel = mag_err.max() / mag_ref.max()
else:
    raise ValueError(f"ORD must be 2 or 'inf', got {ORD!r}")

print(
    f"{name} case {args.case}: relative L{ORD} error {rel:.3e} ({100 * rel:.2f}%)",
    flush=True,
)

# ################## DEBUGGING
# import matplotlib.pyplot as plt

# idx = 0
# fig, ax = plt.subplots(2)
# ax[0].imshow(
#     ref[:, 40, :, idx], vmin=np.min(ref[:, :, :, idx]), vmax=np.max(ref[:, :, :, idx])
# )
# ax[1].imshow(
#     vox[:, 40, :, idx], vmin=np.min(ref[:, :, :, idx]), vmax=np.max(ref[:, :, :, idx])
# )
# plt.show()

# print(ref.shape)
# print(vox.shape)

# print(np.min(ref[:, :, :, idx]), np.max(ref[:, :, :, idx]))
# print(np.min(vox[:, :, :, idx]), np.max(vox[:, :, :, idx]))
