import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cxsp
import cupyx.scipy.sparse.linalg as cxsl
import mlhp
import numpy as np
import scipy.sparse as sp
from pyevtk.hl import imageToVTK

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data/abc"
STL_DIR = DATA_DIR / "geometry/stl"
VOXEL_DIR = DATA_DIR / "geometry/voxel"
FIXTURE_DEF_DIR = DATA_DIR / "fixture_definition"
SOLUTION_DIR = DATA_DIR / "elasticity/solution"
RESULTS_SOLUTION_DIR = BASE_DIR / "../../results/abc/elasticity/solution"

D = 3
VARS = ["x", "y", "z"]

# ----------------------------------- solver settings ---------------------------------
STL_IDS = [4]  # starts at 1; must have a matching loadcase .json
LOADCASE = 0  # which of the 8 load cases to simulate (0-7)

DEGREE = 2
NELEMENTS = 30  # elements along the longest bbox axis; shorter axes scale down to keep cells ~cubic
REFINEMENT = 1  # levels of adaptive refinement toward the geometry boundary (0 = none)
ALPHA_FCM = 1e-4  # fictitious-domain stiffness scaling for cut cells
PENALTY_SCALE = 1e5  # penalty = PENALTY_SCALE * E for the fixed regions
MAXITER = 20000  # gpu iterations are cheap, so allow many
RTOL = 1e-8  # CG relative residual tolerance
JACOBI = True  # diagonal (Jacobi) preconditioner on the gpu; False = no preconditioner

EXPORT_VTU = True  # per-case .pvtu on the STL surface (displacement/stress/von Mises)
EXPORT_VOXEL = True  # sample fields onto the voxel grid of geometry/voxel/<name>.npz
EXPORT_VOXEL_VTU = True  # write the voxelized fields as a .vti for paraview


# ----------------------------------- region helpers ----------------------------------
# Each region (fixed/load) becomes an implicit indicator over (x,y,z); triangles whose
# centroid lies inside are kept, intersected with the grid, and integrated as a surface.
def region_indicator(region, lo, hi):
    kind = region["kind"]
    if kind == "slab":
        a = region["axis"]
        ext = hi[a] - lo[a]
        if region["side"] == "min":
            return f"{VARS[a]} < {lo[a] + region['fraction'] * ext}"
        return f"{VARS[a]} > {hi[a] - region['fraction'] * ext}"
    if kind == "plane":
        p, n, t = region["point"], region["normal"], region["thickness"]
        dot = " + ".join(f"{n[i]}*{VARS[i]}" for i in range(D))
        d = float(np.dot(n, p))
        return f"({dot} - {d})*({dot} - {d}) < {t * t}"
    if kind == "sphere":
        c, r = region["center"], region["radius"]
        sq = " + ".join(f"({VARS[i]} - {c[i]})*({VARS[i]} - {c[i]})" for i in range(D))
        return f"{sq} < {r * r}"
    if kind == "cylinder":
        p, ax, r = region["point"], region["axis"], region["radius"]
        rs = [f"({VARS[i]} - {p[i]})" for i in range(D)]
        proj = " + ".join(f"{ax[i]}*{rs[i]}" for i in range(D))
        dist2 = " + ".join(f"{rs[i]}*{rs[i]}" for i in range(D))
        return f"{dist2} - ({proj})*({proj}) < {r * r}"
    raise ValueError(f"unknown region kind: {kind}")


def region_quadrature(region, surface, grid, lo, hi):
    filtered = surface.filter(
        mlhp.implicitFunction(D, region_indicator(region, lo, hi))
    )
    intersected, celldata = mlhp.intersectWithMesh(filtered, grid)
    return mlhp.simplexQuadrature(intersected, celldata)


def load_integrand(case, lo, hi):
    mag = case["magnitude"]
    if (
        case["type"] == "torsion"
    ):  # tangential traction t = mag * (e_axis x (X - center))
        a = case["load"]["axis"]
        c = 0.5 * (lo + hi)
        rs = [f"({VARS[i]} - {c[i]})" for i in range(D)]
        e = [0.0, 0.0, 0.0]
        e[a] = 1.0
        comps = [
            f"{mag}*({e[1]}*{rs[2]} - {e[2]}*{rs[1]})",
            f"{mag}*({e[2]}*{rs[0]} - {e[0]}*{rs[2]})",
            f"{mag}*({e[0]}*{rs[1]} - {e[1]}*{rs[0]})",
        ]
        return mlhp.neumannIntegrand(
            mlhp.vectorField(D, f"[ {comps[0]}, {comps[1]}, {comps[2]} ]")
        )
    if case["direction"] is None:  # pressure / contact: push along the outward normal
        return mlhp.normalNeumannIntegrand(mlhp.scalarField(D, mag))
    return mlhp.neumannIntegrand(
        mlhp.vectorField(D, [mag * d for d in case["direction"]])
    )


# ----------------------------------- field sampling ----------------------------------
def sample_field(field, points, chunk=200000):
    out = np.empty((len(points), field.odim))
    for i in range(0, len(points), chunk):
        p = points[i : i + chunk]
        out[i : i + chunk] = np.array(field(p[:, 0], p[:, 1], p[:, 2])).reshape(
            -1, field.odim
        )
    return out


def von_mises(stress):  # stress: (N, 9) row-major 3x3 Cauchy tensor
    s = stress.reshape(-1, 3, 3)
    sxx, syy, szz = s[:, 0, 0], s[:, 1, 1], s[:, 2, 2]
    sxy, syz, sxz = s[:, 0, 1], s[:, 1, 2], s[:, 0, 2]
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )


# -------------------------------------- solve ----------------------------------------
for sid in STL_IDS:
    name = f"{sid:09}_abc"
    spec = json.loads((FIXTURE_DEF_DIR / f"{name}.json").read_text())
    case = spec["loadcases"][LOADCASE]

    surface = mlhp.readStl(str(STL_DIR / spec["stl"]))
    kdtree = mlhp.buildKdTree(surface)
    domain = mlhp.rayIntersectionDomain(surface, tree=kdtree)

    E = spec["material"]["E"]
    nu = spec["material"]["nu"]
    penalty = PENALTY_SCALE * E

    lo = np.asarray(spec["bounding_box"]["min"])
    hi = np.asarray(spec["bounding_box"]["max"])
    extent = hi - lo
    h = extent.max() / NELEMENTS  # target (cubic) element size
    nelements = [max(int(np.ceil(extent[d] / h)), 1) for d in range(D)]
    box = np.array(
        [n * h for n in nelements]
    )  # cubic-cell box, >= extent on every axis
    origin = (lo - 0.5 * (box - extent) - 1e-9).tolist()
    lengths = (box + 2e-9).tolist()

    grid = mlhp.makeRefinedGrid(nelements, lengths, origin)
    if REFINEMENT:
        grid.refine(mlhp.refineTowardsBoundary(domain, REFINEMENT))
    basis = mlhp.makeHpTrunkSpace(grid, DEGREE, nfields=D)

    kinematics = mlhp.smallStrainKinematics(D)
    material = mlhp.isotropicElasticMaterial(
        mlhp.scalarField(D, E), mlhp.scalarField(D, nu)
    )
    integrand = mlhp.staticDomainIntegrand(
        kinematics, material, mlhp.vectorField(D, [0.0] * D)
    )
    quadrature = mlhp.momentFittingQuadrature(domain, depth=DEGREE, epsilon=ALPHA_FCM)

    tic = time.perf_counter()
    matrix = mlhp.allocateSparseMatrix(basis)
    vector = mlhp.allocateRhsVector(matrix)
    mlhp.integrateOnDomain(basis, integrand, [matrix, vector], quadrature=quadrature)

    fix_integrand = mlhp.l2BoundaryIntegrand(
        mlhp.vectorField(D, [penalty] * D), mlhp.vectorField(D, [0.0] * D)
    )
    mlhp.integrateOnSurface(
        basis,
        fix_integrand,
        [matrix, vector],
        region_quadrature(case["fixed"], surface, grid, lo, hi),
    )
    mlhp.integrateOnSurface(
        basis,
        load_integrand(case, lo, hi),
        [vector],
        region_quadrature(case["load"], surface, grid, lo, hi),
    )
    print(f"{name}: ndof={basis.ndof()}, assembly {time.perf_counter() - tic:.2f} s", flush=True)

# ----------------------------------- gpu cg solve ------------------------------------
    A = sp.csr_matrix(*matrix.csr_arrays)
    rhs = np.array(vector.tolist(), dtype=np.float64)

    tic = time.perf_counter()
    A_gpu = cxsp.csr_matrix(A)
    b_gpu = cp.asarray(rhs)
    cp.cuda.Stream.null.synchronize()
    transfer = time.perf_counter() - tic

    if JACOBI:
        inv_diag = 1.0 / A_gpu.diagonal()
        M = cxsl.LinearOperator(A_gpu.shape, matvec=lambda x: inv_diag * x)
    else:
        M = None

    iters = [0]
    tic = time.perf_counter()
    x_gpu, info = cxsl.cg(
        A_gpu, b_gpu, tol=RTOL, maxiter=MAXITER, M=M, callback=lambda x: iters.__setitem__(0, iters[0] + 1)
    )
    cp.cuda.Stream.null.synchronize()
    solve = time.perf_counter() - tic

    res = float(cp.linalg.norm(b_gpu - A_gpu @ x_gpu) / cp.linalg.norm(b_gpu))
    dofs = mlhp.DoubleVector(cp.asnumpy(x_gpu).tolist())
    print(
        f"{name} {case['id']}: gpu cg {iters[0]} iters (info={info}), residual {res:.2e}"
        f" | transfer {transfer:.2f} s, solve {solve:.2f} s",
        flush=True,
    )

# ----------------------------------- surface vtu -------------------------------------
    if EXPORT_VTU:
        gradient = mlhp.projectGradient(basis, dofs, quadrature)
        processors = [
            mlhp.solutionProcessor(D, dofs, "Displacement"),
            mlhp.stressProcessor(gradient, kinematics, material),
            mlhp.vonMisesProcessor(gradient, kinematics, material, "VonMises"),
        ]
        intersected, celldata = mlhp.intersectWithMesh(surface, grid, tree=kdtree)
        surfmesh = mlhp.localSimplexCellMesh(intersected, celldata)
        RESULTS_SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
        output = mlhp.PVtuOutput(filename=str(RESULTS_SOLUTION_DIR / f"{name}_{LOADCASE}"))
        mlhp.basisOutput(basis, surfmesh, output, processors)

# ----------------------------------- voxelized fields --------------------------------
    if EXPORT_VOXEL or EXPORT_VOXEL_VTU:
        vox = np.load(VOXEL_DIR / f"{name}.npz")
        indicator = vox["indicator"]
        ncells = indicator.shape
        lengths_v = np.array([float(vox["Lx"]), float(vox["Ly"]), float(vox["Lz"])])
        spacing = lengths_v[0] / ncells[0]  # cubic voxels
        origin_v = lo - 0.5 * (lengths_v - (hi - lo))  # matches voxelization.py

        axes = [origin_v[d] + spacing * (np.arange(ncells[d]) + 0.5) for d in range(D)]
        points = np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])
        mask = indicator.ravel() >= 128  # evaluate only solid voxels

        mech = mlhp.mechanicalEvaluator(basis, dofs, kinematics, material)
        disp = sample_field(mech.displacement, points[mask])
        stress = sample_field(mech.stress, points[mask])

        disp_full = np.zeros((points.shape[0], 3))
        stress_full = np.zeros((points.shape[0], 9))
        vm_full = np.zeros(points.shape[0])
        disp_full[mask] = disp
        stress_full[mask] = stress
        vm_full[mask] = von_mises(stress)

        displacement = disp_full.reshape(*ncells, 3)
        stress_tensor = stress_full.reshape(*ncells, 3, 3)
        vonmises = vm_full.reshape(ncells)
        print(f"\tvoxelized {mask.sum()}/{mask.size} solid voxels", flush=True)

    if EXPORT_VOXEL:
        SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
        out = SOLUTION_DIR / f"{name}_{LOADCASE}.npz"
        np.savez(
            out,
            indicator=indicator,
            displacement=displacement,
            stress=stress_tensor,
            von_mises=vonmises,
            origin=origin_v,
            spacing=spacing,
            Lx=lengths_v[0],
            Ly=lengths_v[1],
            Lz=lengths_v[2],
        )
        print(f"\t-> {out}", flush=True)

    if EXPORT_VOXEL_VTU:
        RESULTS_SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_SOLUTION_DIR / f"{name}_{LOADCASE}_voxel"
        # pyevtk needs contiguous per-component arrays; cell data matches voxel centers
        cell_data = {
            "indicator": np.ascontiguousarray(indicator),
            "von_mises": np.ascontiguousarray(vonmises),
            "displacement": tuple(
                np.ascontiguousarray(displacement[..., i]) for i in range(D)
            ),
        }
        imageToVTK(
            str(out), origin=tuple(origin_v), spacing=(spacing,) * D, cellData=cell_data
        )
        print(f"\t-> {out}.vti", flush=True)
