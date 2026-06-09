import argparse
import json
from pathlib import Path

import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
STL_DIR = DATA_DIR / "geometry/stl"
FIXTURE_DIR = DATA_DIR / "elasticity/fixture"

D = 3

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--loadcase", type=int, default=1)
args = parser.parse_args()

# ----------------------------------- settings ----------------------------------------
E = 210.0
NU = 0.3
MAGNITUDE = 1.0  # nominal traction magnitude
END_FRACTION = 0.1  # grip the skin within this fraction of each major-axis end
OUTWARD_COS_TOL = 0.3  # ... whose outward normal faces along the pull axis (n . axis)

# ----------------------------------- geometry & triangles ----------------------------
name = f"{args.geometry:09}_abc"
surface = mlhp.readStl(str(STL_DIR / f"{name}.stl"))
verts = np.asarray(surface.vertices, dtype=np.float64)
cells = np.asarray(surface.cells, dtype=np.int64)
normals = np.asarray(surface.normals, dtype=np.float64)
centroids = verts[cells].mean(axis=1)

lo, hi = surface.boundingBox()
lo, hi = np.asarray(lo), np.asarray(hi)
extent = hi - lo


# ----------------------------------- load cases --------------------------------------
def tension_major():
    axis = int(np.argmax(extent))
    e = np.zeros(D)
    e[axis] = 1.0
    proj = normals @ e  # signed outward-normal component along the pull axis
    pos = centroids[:, axis]
    near_min = pos < lo[axis] + END_FRACTION * extent[axis]
    near_max = pos > hi[axis] - END_FRACTION * extent[axis]
    return "tension_major", [
        ("fixed", near_min & (proj < -OUTWARD_COS_TOL), "dirichlet", [0.0, 0.0, 0.0]),
        (
            "load",
            near_max & (proj > OUTWARD_COS_TOL),
            "neumann",
            (MAGNITUDE * e).tolist(),
        ),
    ]


CASES = {1: tension_major}

# ----------------------------------- generate ----------------------------------------
(FIXTURE_DIR / "boundary_stl").mkdir(parents=True, exist_ok=True)
label, boundaries = CASES[args.loadcase]()

records = []
for role, tri_mask, btype, value in boundaries:
    if not tri_mask.any():
        raise ValueError(
            f"case {args.loadcase}: boundary '{role}' selected no triangles"
        )
    stl = f"boundary_stl/{name}_{args.loadcase}_{role}.stl"
    mlhp.triangulation(verts, cells[tri_mask]).writeStl(str(FIXTURE_DIR / stl))
    records.append({"name": role, "stl": stl, "type": btype, "value": value})

spec = {
    "geometry": name,
    "stl": f"{name}.stl",
    "case": args.loadcase,
    "label": label,
    "material": {"E": E, "nu": NU, "model": "isotropic"},
    "boundaries": records,
}
path = FIXTURE_DIR / f"{name}_{args.loadcase}.json"
with path.open("w") as f:
    json.dump(spec, f, indent=2)

summary = ", ".join(f"{r}={int(m.sum())} tris" for r, m, _, _ in boundaries)
print(f"{name} case {args.loadcase} ({label}): {summary} -> {path}")
