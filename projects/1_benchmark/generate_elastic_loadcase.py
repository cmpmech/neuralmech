import argparse
import json
from pathlib import Path

import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data/abc"
STL_DIR = DATA_DIR / "geometry/stl"
DEF_DIR = DATA_DIR / "elasticity/fixture_definition"

D = 3

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--case", type=int, default=1)
args = parser.parse_args()

# ----------------------------------- settings ----------------------------------------
E = 210.0
NU = 0.3
MAGNITUDE = 1.0  # nominal traction magnitude
PENALTY_SCALE = 1e5  # penalty = PENALTY_SCALE * E for dirichlet boundaries
END_FRACTION = 0.1  # grip the skin within this fraction of each major-axis end
OUTWARD_COS = 0.3  # ... whose outward normal faces along the pull axis (n . axis)

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
# Each case returns (label, [(role, triangle_mask, type, value), ...]); the masks pick
# the boundary triangles written as standalone STLs and integrated by the solver.
def tension_major():
    axis = int(np.argmax(extent))
    e = np.zeros(D)
    e[axis] = 1.0
    proj = normals @ e  # signed outward-normal component along the pull axis
    pos = centroids[:, axis]
    near_min = pos < lo[axis] + END_FRACTION * extent[axis]
    near_max = pos > hi[axis] - END_FRACTION * extent[axis]
    return "tension_major", [
        ("fixed", near_min & (proj < -OUTWARD_COS), "dirichlet", [0.0, 0.0, 0.0]),
        ("load", near_max & (proj > OUTWARD_COS), "neumann", (MAGNITUDE * e).tolist()),
    ]


CASES = {1: tension_major}

# ----------------------------------- generate ----------------------------------------
DEF_DIR.mkdir(parents=True, exist_ok=True)
label, boundaries = CASES[args.case]()

records = []
for role, tri_mask, btype, value in boundaries:
    if not tri_mask.any():
        raise ValueError(f"case {args.case}: boundary '{role}' selected no triangles")
    stl = f"{name}_{args.case}_{role}.stl"
    mlhp.triangulation(verts, cells[tri_mask]).writeStl(str(DEF_DIR / stl))
    records.append({"name": role, "stl": stl, "type": btype, "value": value})

spec = {
    "geometry": name,
    "stl": f"{name}.stl",
    "case": args.case,
    "label": label,
    "bounding_box": {"min": lo.tolist(), "max": hi.tolist()},
    "material": {"E": E, "nu": NU, "model": "isotropic"},
    "penalty_scale": PENALTY_SCALE,
    "boundaries": records,
}
path = DEF_DIR / f"{name}_{args.case}.json"
with path.open("w") as f:
    json.dump(spec, f, indent=2)

summary = ", ".join(f"{r}={int(m.sum())} tris" for r, m, _, _ in boundaries)
print(f"{name} case {args.case} ({label}): {summary} -> {path}")
