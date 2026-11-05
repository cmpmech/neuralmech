"""engineering tools for the agent example: spec -> topology optimization -> STL -> CAD
interfaces -> re-analysis -> powder bed fusion checks -> report.

Every tool takes and returns JSON strings, writes its files into `WORK_DIR`, and runs its
own checks. A failed check is reported in the returned JSON (`"passed": false` plus the
reason), never hidden; the agent decides what to do next.
"""

import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes
from smolagents import tool

from cuvox.fem.boundary import Dirichlet, assemble
from cuvox.fem.elasticity import Elasticity
from cuvox.fem.materials.elasticity import isotropic
from cuvox.fem.optimization import OptimalityCriterion
from cuvox.fem.regularization import SIMP, DensityFilter
from cuvox.fem.space import Grid, Space, Values

WORK_DIR = Path("agent_runs/current")
E0, NU = 70e3, 0.33  # AlSi10Mg in N/mm^2, the usual powder bed fusion aluminum
EMIN, PENAL, RMIN = 1e-6, 3.0, 2.0
MAX_ITER, CHANGE_TOL = 120, 0.02
OVERHANG = 45.0  # degrees from the build plate that print without support
MIN_WALL = 1.0  # mm, thinnest wall that powder bed fusion builds reliably
LOG = []  # (tool, seconds, passed) of the current run
VALIDATED = {}  # the last spec that passed validate_spec


def _record(name, start, result):
    LOG.append((name, time.time() - start, bool(result.get("passed", True))))
    return json.dumps(result)


def _load(spec):
    h = spec["voxel_size"]
    shape = tuple(int(round(spec[k] / h)) for k in ("length", "width", "height"))
    return spec, h, shape


def _cantilever(shape, h, load):
    """fixed face x=0, line load in -z across the width at mid-height of the free end."""
    space = Space(Grid(shape, h), degree=1, nf=3)
    boundary = ((Dirichlet(), None), (None, None), (None, None))
    fixed = assemble(boundary, cp.ones(shape, cp.bool_), space).fixed
    F = cp.zeros(space.shape, cp.float32)
    weights = cp.ones(space.nodes[1], cp.float32)
    weights[[0, -1]] = 0.5  # trapezoidal weights along the width
    F[2, -1, :, space.nodes[2] // 2] = -load * weights / weights.sum()
    return space, fixed, F


def _solve(space, fixed, F, stiffness):
    model = Elasticity(space, Values(stiffness), isotropic(NU), fixed=fixed)
    u, _ = model.solve(F, etol=1e-6)
    return float(cp.vdot(F, u)) / E0, u, model


# ---------------------------------------- tools --------------------------------------
@tool
def validate_spec(spec: dict) -> str:
    """Checks a design specification and returns it normalized. Call this first.

    Args:
        spec: object with "length", "width", "height" (design envelope in mm),
            "load" (N, downward at mid-height of the free end, fixed face at x=0),
            "volume_fraction" (0.05 to 0.5), "voxel_size" (mm), "build_direction" ("z").
    """
    start = time.time()
    problems = []
    spec = dict(spec)
    keys = ["length", "width", "height", "load", "volume_fraction", "voxel_size"]
    problems += [f"missing {k}" for k in keys if k not in spec]
    if not problems:
        if min(spec["length"], spec["width"], spec["height"], spec["voxel_size"]) <= 0:
            problems.append("lengths and voxel size must be positive")
        if not 0.05 <= spec["volume_fraction"] <= 0.5:
            problems.append("volume_fraction must lie in [0.05, 0.5]")
        if spec["load"] == 0:
            problems.append("load must not be zero")
        spec.setdefault("build_direction", "z")
        if spec["build_direction"] != "z":
            problems.append("only build_direction z is supported")
        if not problems:
            _, _, shape = _load(spec)
            if min(shape) < 8:
                problems.append(f"too coarse, {min(shape)} voxels across the thinnest side (8 needed)")
            if np.prod(shape) > 400_000:
                problems.append(f"too fine, {np.prod(shape)} voxels (400000 allowed)")
    result = {"passed": not problems, "problems": problems, "spec": spec}
    if not problems:
        VALIDATED["spec"] = spec
    return _record("validate_spec", start, result)


@tool
def topology_optimization(spec: dict) -> str:
    """Minimizes the compliance of the design envelope for the given volume fraction (SIMP,
    optimality criteria, GPU). Returns the density file and the optimized compliance.

    Args:
        spec: the normalized spec returned by validate_spec.
    """
    start = time.time()
    spec, h, shape = _load(spec)
    space, fixed, F = _cantilever(shape, h, spec["load"])
    x = cp.full(shape, spec["volume_fraction"], cp.float32)
    density_filter = DensityFilter(RMIN, shape)
    dv = density_filter.grad(x, cp.ones_like(x))
    simp = SIMP(PENAL, EMIN)
    optimizer = OptimalityCriterion(spec["volume_fraction"], move=0.1, damping=0.5)
    model = Elasticity(space, Values(simp(x)), isotropic(NU), fixed=fixed)
    u, change = None, 1.0
    for it in range(MAX_ITER):
        rho = density_filter(x)
        model.update(Values(simp(rho)), bounds=it % 10 == 0)
        u, _ = model.solve(F, x0=u, etol=1e-6)
        dc = simp.grad(rho, model.vjp(u, u))
        x_new = optimizer.step(x, density_filter.grad(x, dc), dv)
        change = float(cp.abs(x_new - x).max())
        x = x_new
        if change < CHANGE_TOL:
            break
    rho = density_filter(x).get()
    compliance = float(cp.vdot(F, u)) / E0
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    np.save(WORK_DIR / "density.npy", rho)
    gray = float(np.mean((rho > 0.1) & (rho < 0.9)))
    volume = float(rho.mean())
    problems = []
    if abs(volume - spec["volume_fraction"]) > 0.01:
        problems.append(f"volume fraction {volume:.3f} misses the target")
    result = {"passed": not problems, "problems": problems, "density_file": str(WORK_DIR / "density.npy"),
              "compliance_Nmm": compliance, "iterations": it + 1, "converged": change < CHANGE_TOL,
              "gray_fraction": gray, "volume_fraction": volume}
    return _record("topology_optimization", start, result)


@tool
def extract_stl(density_file: str, threshold: float, voxel_size: float) -> str:
    """Thresholds the density and extracts a triangle surface (marching cubes) as an STL file.

    Args:
        density_file: the .npy file from topology_optimization.
        threshold: density above which a voxel is solid, typically 0.5.
        voxel_size: the voxel size in mm from the spec.
    """
    start = time.time()
    solid = np.load(density_file) > threshold
    labels, count = ndimage.label(solid)
    sizes = ndimage.sum(solid, labels, range(1, count + 1)) if count else np.array([0])
    problems = []
    if count == 0:
        return _record("extract_stl", start, {"passed": False, "problems": ["no solid voxels"]})
    if count > 1:
        problems.append(f"{count} disconnected parts, the largest holds {sizes.max() / sizes.sum():.1%}")
    np.save(WORK_DIR / "solid.npy", solid)
    mesh = _surface(solid, voxel_size)
    mesh.export(WORK_DIR / "design.stl")
    voxel_volume = solid.sum() * voxel_size**3
    if not mesh.is_watertight:
        problems.append("surface is not watertight")
    elif abs(mesh.volume / voxel_volume - 1) > 0.1:
        problems.append(f"STL volume {mesh.volume:.0f} mm^3 differs from the voxel volume {voxel_volume:.0f} mm^3")
    result = {"passed": not problems, "problems": problems, "solid_file": str(WORK_DIR / "solid.npy"),
              "stl_file": str(WORK_DIR / "design.stl"), "watertight": bool(mesh.is_watertight),
              "stl_volume_mm3": float(mesh.volume), "voxel_volume_mm3": float(voxel_volume)}
    return _record("extract_stl", start, result)


def _surface(solid, h):
    padded = np.pad(solid.astype(np.float32), 1)
    vertices, faces, _, _ = marching_cubes(padded, 0.5, spacing=(h, h, h))
    mesh = trimesh.Trimesh(vertices - h / 2, faces[:, ::-1])  # voxel faces at multiples of h
    trimesh.repair.fix_normals(mesh)
    return mesh


@tool
def add_interfaces(solid_file: str, spec: dict) -> str:
    """Builds the mounting plate with two bolt holes at the fixed face and a load pad at the
    free end in CAD (build123d), adds them to the design, keeps the bolt holes free, and
    checks that the part stays inside the design envelope. Writes STEP and STL.

    Args:
        solid_file: the .npy file of solid voxels from extract_stl.
        spec: the normalized spec.
    """
    from build123d import Box, Cylinder, Location, Rotation, export_step, export_stl

    start = time.time()
    spec, h, shape = _load(spec)
    L, W, H = spec["length"], spec["width"], spec["height"]
    t = 2 * h  # plate and pad thickness
    plate = Location((t / 2, W / 2, H / 2)) * Box(t, W, H)
    holes = [Location((t / 2, W / 2, z)) * Rotation(0, 90, 0) * Cylinder(min(W, H) / 8, 2 * t) for z in (H / 4, 3 * H / 4)]
    for hole in holes:
        plate = plate - hole
    pad = Location((L - t / 2, W / 2, H / 2)) * Box(t, W, H / 4)
    interfaces = plate + pad
    export_step(interfaces, str(WORK_DIR / "interfaces.step"))
    export_stl(interfaces, str(WORK_DIR / "interfaces.stl"))

    # voxel centers inside the CAD solids
    centers = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) * h for n in shape], indexing="ij"), -1).reshape(-1, 3)
    inside = trimesh.load(WORK_DIR / "interfaces.stl").contains(centers).reshape(shape)
    keep_out = np.zeros(shape, bool)
    for z in (H / 4, 3 * H / 4):
        r2 = (centers[:, 1] - W / 2) ** 2 + (centers[:, 2] - z) ** 2
        keep_out |= ((r2 <= (min(W, H) / 8) ** 2) & (centers[:, 0] <= 2 * t)).reshape(shape)
    part = (np.load(solid_file) | inside) & ~keep_out
    np.save(WORK_DIR / "part.npy", part)
    mesh = _surface(part, h)
    mesh.export(WORK_DIR / "part.stl")
    lower, upper = mesh.bounds
    problems = []
    if (lower < -1e-6).any() or (upper > np.array([L, W, H]) + 1e-6).any():
        problems.append("part leaves the design envelope")
    if part[keep_out].any():
        problems.append("material inside the bolt holes")
    result = {"passed": not problems, "problems": problems, "part_file": str(WORK_DIR / "part.npy"),
              "stl_file": str(WORK_DIR / "part.stl"), "step_file": str(WORK_DIR / "interfaces.step"),
              "bounds_mm": [lower.tolist(), upper.tolist()]}
    return _record("add_interfaces", start, result)


@tool
def reanalyze(part_file: str, spec: dict, optimized_compliance: float) -> str:
    """Solves the final black-and-white part with the finite element method and compares its
    compliance with the optimized one (at most 25 % more compliant passes).

    Args:
        part_file: the .npy file of the final part from add_interfaces.
        spec: the normalized spec.
        optimized_compliance: compliance_Nmm returned by topology_optimization.
    """
    start = time.time()
    spec, h, shape = _load(spec)
    part = np.load(part_file)
    space, fixed, F = _cantilever(shape, h, spec["load"])
    stiffness = cp.asarray(np.where(part, 1.0, EMIN), cp.float32)
    compliance, u, _ = _solve(space, fixed, F, stiffness)
    ratio = compliance / optimized_compliance
    tip = float(cp.abs(u[2, -1]).max()) / E0
    problems = [] if ratio <= 1.25 else [f"compliance {ratio:.2f} times the optimized one"]
    result = {"passed": not problems, "problems": problems, "compliance_Nmm": compliance,
              "ratio_to_optimized": ratio, "max_tip_displacement_mm": tip}
    return _record("reanalyze", start, result)


@tool
def pbf_checks(part_file: str, voxel_size: float) -> str:
    """Geometric checks for powder bed fusion with build direction z: enclosed powder
    (empty regions without connection to the outside), walls thinner than 1 mm, and
    the area of downward surfaces flatter than 45 degrees that needs support.

    Args:
        part_file: the .npy file of the final part from add_interfaces.
        voxel_size: the voxel size in mm from the spec.
    """
    start = time.time()
    part = np.load(part_file)
    empty, count = ndimage.label(np.pad(~part, 1, constant_values=True))
    trapped = int(np.sum(empty[1:-1, 1:-1, 1:-1] > 1)) if count > 1 else 0
    padded = np.pad(part, 1)
    single = np.zeros_like(part)
    for axis in range(3):  # empty on both sides along one axis
        before = np.take(padded, range(0, part.shape[axis]), axis=axis)
        after = np.take(padded, range(2, part.shape[axis] + 2), axis=axis)
        sl = [slice(1, -1)] * 3
        sl[axis] = slice(None)
        single |= ~before[tuple(sl)] & ~after[tuple(sl)]
    thin = float((part & single).sum() / part.sum()) if voxel_size < MIN_WALL else 0.0
    mesh = _surface(part, voxel_size)
    down = mesh.face_normals[:, 2] < -np.cos(np.radians(OVERHANG))
    above_plate = mesh.triangles_center[:, 2] > voxel_size
    support = float(mesh.area_faces[down & above_plate].sum())
    problems = []
    if trapped:
        problems.append(f"{trapped} voxels of enclosed powder")
    if thin > 0.02:
        problems.append(f"{thin:.1%} of the material in walls thinner than {MIN_WALL} mm")
    result = {"passed": not problems, "problems": problems, "trapped_powder_voxels": trapped,
              "thin_fraction": thin, "support_area_mm2": support,
              "support_area_fraction": support / float(mesh.area)}
    return _record("pbf_checks", start, result)


@tool
def write_report(summary: dict) -> str:
    """Writes the final report. Call this last, with every check result, also when the
    specification failed.

    Args:
        summary: object with the spec, the files, and the result of every tool.
    """
    start = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "report.json").write_text(json.dumps(summary, default=str))
    return _record("write_report", start, {"passed": True, "report_file": str(WORK_DIR / "report.json")})


TOOLS = [validate_spec, topology_optimization, extract_stl, add_interfaces, reanalyze, pbf_checks, write_report]
