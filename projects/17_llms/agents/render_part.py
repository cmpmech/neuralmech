import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../../results").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--run", default="gpt-oss-20b/task0_run0")
parser.add_argument("--voxel_size", type=float, default=2.5)  # mm, of task 0
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
OVERHANG = 45.0  # degrees, as in helper.py
LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
VIEWS = [(22, -58), (-28, -58)]  # elevation, azimuth: design from above, part from below


# --------------------------------------- helper --------------------------------------
def surface(solid, h):
    vertices, faces, _, _ = marching_cubes(np.pad(solid.astype(float), 1), 0.5, spacing=(h, h, h))
    triangles = (vertices - h / 2)[faces[:, ::-1]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return triangles, normals / np.linalg.norm(normals, axis=1, keepdims=True)


def draw(ax, triangles, normals, view, highlight=None):
    light = LIGHT * np.array([1, 1, np.sign(view[0])])  # light from the side of the viewer
    shade = 0.35 + 0.65 * np.clip(normals @ light, 0, 1)
    colors = np.tile([0.62, 0.66, 0.74], (len(triangles), 1)) * shade[:, None]
    if highlight is not None:
        colors[highlight] = np.array([0.85, 0.45, 0.35]) * shade[highlight, None]
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, linewidths=0))
    lower, upper = triangles.reshape(-1, 3).min(0), triangles.reshape(-1, 3).max(0)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(upper - lower)
    ax.view_init(*view)
    ax.set_axis_off()


# ------------------------------------- load data -------------------------------------
run = RESULTS_DIR / "agent_runs" / args.run
design = surface(np.load(run / "solid.npy"), args.voxel_size)
part = surface(np.load(run / "part.npy"), args.voxel_size)
down = part[1][:, 2] < -np.cos(np.radians(OVERHANG))
support = down & (part[0][:, :, 2].mean(1) > args.voxel_size)  # not on the build plate

# ----------------------------------- postprocessing ----------------------------------
fig = plt.figure(figsize=(8, 3))
for i, (mesh, highlight) in enumerate([(design, None), (part, support)]):
    draw(fig.add_subplot(1, 2, i + 1, projection="3d"), *mesh, VIEWS[i], highlight)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    plt.savefig(RGB_PNG_DIR / "agent_part.png", dpi=250)
