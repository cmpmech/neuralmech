import json
from pathlib import Path

import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data/abc"
STL_DIR = DATA_DIR / "geometry/stl"
FIXTURE_DEF_DIR = DATA_DIR / "fixture_definition"

# --------------------------------- load-case settings --------------------------------
STL_IDS = [1, 3, 4]  # starts at 1

GRIP_FRACTION = 0.05  # slab thickness as fraction of the axis extent
PATCH_RADIUS_FRAC = 0.1  # sphere-patch radius as fraction of the bounding diagonal

# flat-face clustering: bin triangle normals by direction and signed plane offset
FLATFACE_NORMAL_TOL = 0.08
FLATFACE_OFFSET_FRAC = 0.02
MIN_FLATFACE_AREA_FRAC = 0.05  # a flat face must hold this share of the total area

# best-effort cylindrical-hole detection (least-squares axis fit per principal axis)
DETECT_HOLES = True
HOLE_AXIS_PERP_TOL = 0.25  # |normal . axis| below this counts as side-of-cylinder
HOLE_MIN_TRIANGLES = 12
HOLE_MAX_RADIUS_FRAC = 0.4
HOLE_RADIUS_REL_STD = 0.25  # accept only if radii are tight around the mean

E = 210.0
NU = 0.3
MAGNITUDE = 1.0  # nominal traction / pressure; solver may renormalize by grip area

SKIP_DISCONNECTED = True
FAILURE_LOG = DATA_DIR / "disconnected.txt"

# --------------------------------- region & feature helpers ---------------------------
def slab(axis, side, fraction):
    return {"kind": "slab", "axis": int(axis), "side": side, "fraction": float(fraction)}


def axis_dir(axis, sign):
    d = [0.0, 0.0, 0.0]
    d[axis] = float(sign)
    return d


def make_case(cid, ctype, fixed, load, direction, mechanism):
    return {
        "id": cid,
        "type": ctype,
        "fixed": fixed,
        "load": load,
        "direction": None if direction is None else [float(v) for v in direction],
        "magnitude": MAGNITUDE,
        "bc": "penalty",
        "mechanism": mechanism,
    }


def triangle_data(surface):
    verts = np.asarray(surface.vertices, dtype=np.float64)
    cells = np.asarray(surface.cells, dtype=np.int64)
    tri = verts[cells]  # (Nc, 3, 3)
    centroids = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)
    return centroids, areas, normals


def largest_flat_face(centroids, areas, normals, total_area, diag):
    offset_bin = max(FLATFACE_OFFSET_FRAC * diag, 1e-12)
    nbin = np.round(normals / max(FLATFACE_NORMAL_TOL, 1e-6)).astype(np.int64)
    dbin = np.round(np.sum(normals * centroids, axis=1) / offset_bin).astype(np.int64)
    key = np.column_stack([nbin, dbin[:, None]])
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    bucket_area = np.zeros(len(uniq))
    np.add.at(bucket_area, inv, areas)
    best = int(np.argmax(bucket_area))
    if bucket_area[best] < MIN_FLATFACE_AREA_FRAC * total_area:
        return None
    mask = inv == best
    w = areas[mask]
    p = np.average(centroids[mask], axis=0, weights=w)
    n = np.average(normals[mask], axis=0, weights=w)
    n = n / (np.linalg.norm(n) + 1e-30)
    return {"kind": "plane", "point": p.tolist(), "normal": n.tolist(),
            "thickness": float(offset_bin)}


def detect_hole(centroids, areas, normals, diag):
    if not DETECT_HOLES:
        return None
    best = None
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        side = np.abs(normals @ e) < HOLE_AXIS_PERP_TOL
        if int(side.sum()) < HOLE_MIN_TRIANGLES:
            continue
        c = centroids[side] - np.outer(centroids[side] @ e, e)  # in-plane centroids
        n = normals[side] - np.outer(normals[side] @ e, e)  # in-plane normals
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
        # axis point p minimizing summed squared distance to the lines {c_i + t n_i}
        nt = len(c)
        A = nt * np.eye(3) - n.T @ n + np.outer(e, e)  # axis component pinned to 0
        b = c.sum(axis=0) - (n * np.sum(n * c, axis=1, keepdims=True)).sum(axis=0)
        try:
            p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        radii = np.linalg.norm(np.cross(c - p, n), axis=1)
        rmean = float(radii.mean())
        if rmean <= 0 or rmean > HOLE_MAX_RADIUS_FRAC * diag:
            continue
        if radii.std() / rmean > HOLE_RADIUS_REL_STD:
            continue
        if best is None or nt > best[0]:
            point = p + float((centroids[side] @ e).mean()) * e
            best = (nt, {"kind": "cylinder", "point": point.tolist(),
                         "axis": e.tolist(), "radius": rmean})
    return None if best is None else best[1]


def sphere_patch(centroids, target, diag):
    i = int(np.argmin(np.linalg.norm(centroids - target, axis=1)))
    return {"kind": "sphere", "center": centroids[i].tolist(),
            "radius": float(PATCH_RADIUS_FRAC * diag)}


def build_loadcases(long_, mid, short, flat_face, hole, sphere):
    grip_min = slab(long_, "min", GRIP_FRACTION)
    grip_max = slab(long_, "max", GRIP_FRACTION)
    cases = [
        make_case("tension_long", "tension", grip_min, grip_max,
                  axis_dir(long_, +1), "slab"),
        make_case("compression_long", "compression", grip_min, grip_max,
                  axis_dir(long_, -1), "slab"),
        make_case("bending_long", "bending", grip_min, grip_max,
                  axis_dir(short, +1), "slab"),
        make_case("torsion_long", "torsion", grip_min, grip_max, None, "slab"),
        make_case("crossaxis", "shear", grip_min, slab(mid, "max", GRIP_FRACTION),
                  axis_dir(mid, +1), "cross_axis"),
    ]

    if flat_face is not None:
        cases.append(make_case("pressure_face", "pressure", grip_min, flat_face,
                               None, "flat_face"))
        cases.append(make_case("mount_face", "tension", flat_face, grip_max,
                               axis_dir(long_, +1), "flat_face"))
    else:
        cases.append(make_case("pressure_face", "pressure", grip_min, grip_max,
                               None, "slab"))
        cases.append(make_case("mount_face", "tension", grip_min, grip_max,
                               axis_dir(long_, +1), "slab"))

    if hole is not None:
        cases.append(make_case("contact_local", "contact", hole, grip_max,
                               axis_dir(long_, +1), "cylinder"))
    elif sphere is not None:
        cases.append(make_case("contact_local", "contact", grip_min, sphere,
                               None, "sphere"))
    else:
        cases.append(make_case("contact_local", "shear", grip_min, grip_max,
                               axis_dir(mid, +1), "slab"))
    return cases


# ------------------------------------- generation ------------------------------------
FIXTURE_DEF_DIR.mkdir(parents=True, exist_ok=True)
skip = set()
if SKIP_DISCONNECTED and FAILURE_LOG.exists():
    skip = set(FAILURE_LOG.read_text().split())

for STL_IDX in STL_IDS:
    name = f"{STL_IDX:09}_abc"
    if name in skip:
        print(f"{name}: skipped (disconnected)")
        continue

    surface = mlhp.readStl(str(STL_DIR / f"{name}.stl"))
    centroids, areas, normals = triangle_data(surface)

    keep = areas > 1e-18  # drop degenerate triangles before feature detection
    centroids, areas, normals = centroids[keep], areas[keep], normals[keep]

    (lo, hi) = surface.boundingBox()
    lo, hi = np.asarray(lo), np.asarray(hi)
    extent = hi - lo
    diag = float(np.linalg.norm(extent))
    order = np.argsort(extent)
    short, mid, long_ = int(order[0]), int(order[1]), int(order[2])

    total_area = float(areas.sum())
    flat_face = largest_flat_face(centroids, areas, normals, total_area, diag)
    hole = detect_hole(centroids, areas, normals, diag)
    target = np.asarray(flat_face["point"]) if flat_face else np.average(
        centroids, axis=0, weights=areas
    )
    sphere = sphere_patch(centroids, target, diag)

    cases = build_loadcases(long_, mid, short, flat_face, hole, sphere)

    out = {
        "stl": f"{name}.stl",
        "bounding_box": {"min": lo.tolist(), "max": hi.tolist()},
        "material": {"E": E, "nu": NU, "model": "isotropic"},
        "loadcases": cases,
    }
    path = FIXTURE_DEF_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(out, f, indent=2)

    feats = [m for m, v in (("flat_face", flat_face), ("hole", hole)) if v]
    print(
        f"{name}: {len(cases)} cases, {len(centroids)} triangles,"
        f" features={feats or ['none']} -> {path}"
    )
