# This file is part of the scanbasedanalysis project. License: See LICENSE

"""Phase-field fracture shear test, ported from examples/sheartest.cpp (which
relied on the C++ helpers in include/fracture/*.hpp) to pure Python on top of
mlhp. All phase-field functionality lives at the top of this single file: the
three "user subroutines" (degraded material, history update, phase-field
residual/tangent) are written in C and compiled on the fly with cffi, then
passed to mlhp as function pointers.

Model: AT2-type brittle phase field with a tension-only (volumetric split)
history driving force, staggered solution (mechanical solve -> history update
-> phase-field Newton) and phase-field-driven adaptive mesh refinement around
the crack. 2D plane-stress single-edge-notched specimen sheared at the top.

The defaults below use a COARSE discretization for a fast first run; the
original sheartest.cpp values are given as inline comments so they can be
swapped back in for a production run.

Backend note: cffi compiles the C below with the system C compiler (MSVC on
Windows, located automatically through setuptools). If you prefer not to
compile, the same callbacks can be written with numba (@cfunc) as in mlhp's
poisson_compiled.py example.
"""

import math
import os
import sys

import cffi
import numpy as np

import mlhp

D = 2  # 2D for now; the C callbacks below are written for plane stress (3 Voigt
       # components). A 3D extension would generalize them to 6 components.

# ---------------------------------------------------------------------------
# Numerical / physical parameters
# ---------------------------------------------------------------------------

# Material and phase-field parameters (same as sheartest.cpp)
E = 210.0           # Young's modulus
nu = 0.3            # Poisson ratio
baseGc = 0.0027     # critical energy release rate Gc
l0 = 0.008          # phase-field length scale
eta = 1e-6          # residual stiffness / degradation floor

# Discretization -- COARSE defaults for a quick run.
nelements = 16      # per direction.        sheartest.cpp: 33
degree = 1          # polynomial degree.    sheartest.cpp: 1
nsteps = 15         # load steps.           sheartest.cpp: 50
length = 1.0        # square domain edge.   sheartest.cpp: 1.0
uxMax = 0.018       # max top displacement. sheartest.cpp: 0.018

# Phase-field refinement thresholds: a cell is refined to level k when the phase
# field there drops below thresholds[k] (replicates refineWithPhaseField).
refinementThresholds = [0.95, 0.8, 0.5]

# Staggered loop control
maxStaggered = 30   # safety cap (sheartest.cpp loops until converged)
staggeredTol = 1e-3 # relative change of the phase field between staggered iters
maxNewton = 20      # phase-field Newton iterations per staggered iteration

# The C-extension was written against this mlhp C-ABI version. mlhp checks it at
# registration; if your mlhp build bumps the ABI, update the callbacks to the new
# signatures (see the factory docstrings) and increment this literal.
cabi = 1

outputDir = "outputs"
os.makedirs(outputDir, exist_ok=True)

# ===========================================================================
# Phase-field user subroutines, compiled with cffi
# ===========================================================================
#
# Voigt convention (2D plane stress): ncomponents = 3, order [xx, yy, xy] with
# engineering shear strain. The plane-stress matrix is
#     C = E/(1-nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
# and degradation g(s) = (1-eta) * clamp(s, eta, 1)^2 + eta (defaultDegradation).

print("Compiling phase-field C callbacks (cffi)...", flush=True)

ffibuilder = cffi.FFI()
ffibuilder.cdef(
    "extern const unsigned long long material_address, update_address, "
    "phasefield_address;")
ffibuilder.set_source("_sheartest_cffi", r"""
    #include <math.h>
    #include <stdint.h>

    /* ---- degraded constitutive equation -------------------------------- *
     * Signature: mlhp.constitutiveEquation. The phase field s is supplied as
     * user field 0 (a mesh function wrapping the phase-field solution); the
     * material parameters [E, nu, eta] come in as data vector 0. Computes the
     * plane-stress stress/tangent and scales them by g(s).                  */
    static int64_t material(double* stress, double* tangent, double* energyDensity,
                            double* gradient, double* strain, double* xyz, double* rst,
                            double** userFields, double** userData, double* tmp,
                            int64_t* sizes, int64_t* userFieldSizes, int64_t ielement)
    {
        double E = userData[0][0], nu = userData[0][1], eta = userData[0][2];

        double s = userFields[0][0];
        double sc = s < eta ? eta : (s > 1.0 ? 1.0 : s);
        double g = (1.0 - eta) * sc * sc + eta;

        double f = E / (1.0 - nu * nu);
        double C[9] = { f, f * nu, 0.0, f * nu, f, 0.0, 0.0, 0.0, f * (1.0 - nu) / 2.0 };

        /* mlhp calls this in different modes: in the matrix-only (tangent)
         * path strain/stress are null, so only touch strain when it is given. */
        if(strain)
        {
            double st[3];
            st[0] = C[0] * strain[0] + C[1] * strain[1];
            st[1] = C[3] * strain[0] + C[4] * strain[1];
            st[2] = C[8] * strain[2];

            if(stress) { for(int i = 0; i < 3; ++i) stress[i] = g * st[i]; }
            if(energyDensity)
                energyDensity[0] = 0.5 * g * (st[0] * strain[0] + st[1] * strain[1] +
                                              st[2] * strain[2]);
        }
        if(tangent) { for(int i = 0; i < 9; ++i) tangent[i] = g * C[i]; }
        return 0;
    }

    /* ---- history update (max tensile strain energy) -------------------- *
     * Signature: mlhp.meshFunctionStrainUpdate. values[0] holds the old H and
     * is overwritten with max(H, psi+). psi+ is the tensile strain energy: the
     * compressive volumetric part is removed when the volumetric strain < 0.
     * Only the current state (strains[1]) is needed. data vector 0 = [E, nu]. */
    static int64_t update(double* values, double** gradients, double** strains,
                          double* xyz, double* rst, double** userFields,
                          double** userData, double* tmp, int64_t* sizes,
                          int64_t* userFieldSizes, int64_t ielement)
    {
        double E = userData[0][0], nu = userData[0][1];

        double e[3] = { strains[1][0], strains[1][1], strains[1][2] };
        double vol = e[0] + e[1];

        if(vol < 0.0) { e[0] -= 0.5 * vol; e[1] -= 0.5 * vol; }  /* 1/D, D = 2 */

        double f = E / (1.0 - nu * nu);
        double st0 = f * (e[0] + nu * e[1]);
        double st1 = f * (nu * e[0] + e[1]);
        double st2 = f * (1.0 - nu) / 2.0 * e[2];

        double Estr = 0.5 * (st0 * e[0] + st1 * e[1] + st2 * e[2]);

        if(Estr > values[0]) values[0] = Estr;
        return 0;
    }

    /* ---- phase-field residual / tangent integrand --------------------- *
     * Signature: mlhp.domainIntegrand. Targets: [matrix, vector]. The current
     * phase-field dofs come in as data vector 0 (indexed through locationMap),
     * the history H as user field 0, and [Gc, l0, eta] as data vector 1.
     * Direct transcription of makePhaseFieldIntegrand.                      */
    static int64_t phasefield(double** targets, double** shapes, double** mapping,
                              double* rst, double** userFields, double** userData,
                              double* tmp, int64_t* locationMap, int64_t* sizes,
                              int64_t* shapeSizes, int64_t* userFieldSizes,
                              double weight, int64_t ielement)
    {
        int64_t ndof = sizes[3], ndofpadded = sizes[4];

        double* N   = shapes[0];
        double* dNx = shapes[0] + ndofpadded;
        double* dNy = shapes[0] + 2 * ndofpadded;

        double* pdofs = userData[0];
        double Gc = userData[1][0], l0 = userData[1][1], eta = userData[1][2];
        double H = userFields[0][0];

        double s = 0.0, dsx = 0.0, dsy = 0.0;
        for(int64_t i = 0; i < ndof; ++i)
        {
            double d = pdofs[locationMap[i]];
            s += N[i] * d; dsx += dNx[i] * d; dsy += dNy[i] * d;
        }

        double sc = s < eta ? eta : s;          /* defaultDegradation clamp */
        double dg = 2.0 * (1.0 - eta) * sc;
        double ddg = 2.0 * (1.0 - eta);

        double MM1 = 2.0 * l0 / Gc * H * dg + s;
        double MM2 = 2.0 * (l0 / Gc) * H * ddg + 1.0;
        double ST = 4.0 * l0 * l0;

        double* K = targets[0];
        double* F = targets[1];

        for(int64_t i = 0; i < ndof; ++i)
        {
            for(int64_t j = 0; j < ndof; ++j)
                K[i * ndofpadded + j] += (N[i] * N[j] * MM2 +
                    dNx[i] * dNx[j] * ST + dNy[i] * dNy[j] * ST) * weight;

            F[i] += -(N[i] * (MM1 - 1.0) + (dNx[i] * dsx + dNy[i] * dsy) * ST) * weight;
        }
        return 0;
    }

    const unsigned long long material_address   = (unsigned long long)&material;
    const unsigned long long update_address     = (unsigned long long)&update;
    const unsigned long long phasefield_address = (unsigned long long)&phasefield;
""")

ffibuilder.compile(tmpdir=outputDir)
sys.path.insert(0, outputDir)
from _sheartest_cffi import lib  # noqa: E402

# ===========================================================================
# Preprocessing
# ===========================================================================

print("Preprocessing...", flush=True)

lengths = [length] * D
matParams = [E, nu, eta]          # for the degraded material callback
elasticParams = [E, nu]           # for the history update callback
pfParams = [baseGc, l0, eta]      # for the phase-field integrand callback

kinematics = mlhp.smallStrainKinematics(D)
zeroBody = mlhp.vectorField(D, [0.0] * D)

# Initial crack: a thin notch from the left edge to the centre, at mid-height.
x0, x1 = 0.0, length / 2.0
y0, y1 = length / 2.0 - l0, length / 2.0 + l0
crack = mlhp.implicitCube((x0, y0), (x1, y1))

baseGrid = mlhp.makeGrid([nelements] * D, lengths)

# Initial mesh: refined inside the crack, as many levels as thresholds.
grid0 = mlhp.makeRefinedGrid(baseGrid)
grid0.refine(mlhp.refineInsideDomain(crack, len(refinementThresholds), degree + 2))

# Phase field initialised to 1 (intact) everywhere.
pbasis0 = mlhp.makeHpTrunkSpace(grid0, degree=degree, nfields=1)
pdofs0 = mlhp.projectOnto(pbasis0, mlhp.scalarField(D, 1.0))

# History initialised large inside the crack to seed it, 0 elsewhere.
Hinit = 1000.0 * baseGc / (4.0 * l0)
initExpr = (f"{Hinit} * (x >= {x0}) * (x <= {x1}) * "
            f"(y >= {y0}) * (y <= {y1})")
history0 = mlhp.meshFunction(grid0, mlhp.vectorField(D, f"[{initExpr}]"))
pmesh0 = grid0

# Refinement level field: target level = number of thresholds the phase field
# falls below, evaluated through the current phase-field solution (= f0).
levelExpr = "+".join(f"(f0(x,y)<={t})" for t in refinementThresholds)

print(f"Initial state: {pbasis0.nelements()} elements, "
      f"{pbasis0.ndof()} phase-field dofs", flush=True)

# Reaction-force output file (overwrite + header)
csvPath = os.path.join(outputDir, "sheartest.csv")
with open(csvPath, "w") as f:
    f.write("ux;Rx;Ry\n")


def degradedMaterial(pbasis, pdofs):
    """Linear-elastic plane-stress material degraded by the given phase field."""
    return mlhp.constitutiveEquation(
        D, lib.material_address, voigt=True, symmetric=True,
        fields=[mlhp.meshFunction(pbasis, pdofs)], data=[matParams], abi=cabi)


# ===========================================================================
# Load stepping
# ===========================================================================

filebase = os.path.join(outputDir, f"sheartest_{D}D_")

for istep in range(nsteps + 1):
    ux = istep * uxMax / nsteps

    for istaggered in range(maxStaggered):
        print(f"Load step {istep} / {nsteps}, staggered iteration "
              f"{istaggered + 1}", flush=True)

        # -- (Re)build mesh from the previous phase field: refine around the
        #    crack and never coarsen below the previous refinement. ---------
        grid1 = mlhp.makeRefinedGrid(baseGrid)
        levelField = mlhp.scalarField(
            D, levelExpr, fields=[mlhp.scalarEvaluator(pbasis0, pdofs0)])
        refinePhase = mlhp.refineWithLevelFunction(levelField, degree + 2)
        refineKeep = mlhp.refineAdaptively(pmesh0, [0] * pmesh0.ncells())
        grid1.refine(mlhp.refinementOr([refinePhase, refineKeep]))

        mbasis1 = mlhp.makeHpTrunkSpace(grid1, degree=degree, nfields=D)
        pbasis1 = mlhp.makeHpTrunkSpace(grid1, degree=degree, nfields=1)

        # Project the previous phase field onto the new mesh (initial guess and
        # degradation field for the mechanical solve).
        pdofs0g1 = mlhp.projectOnto(
            pbasis1, mlhp.scalarEvaluator(pbasis0, pdofs0))

        # -- Mechanical solve (phase field held fixed) ---------------------
        dirTop = mlhp.integrateDirichletDofs(
            mlhp.vectorField(D, [ux, 0.0]), mbasis1, [3])   # face 3 = ymax (top)
        dirBot = mlhp.integrateDirichletDofs(
            mlhp.vectorField(D, [0.0, 0.0]), mbasis1, [2])  # face 2 = ymin (bot)
        dirichlet = mlhp.combineDirichletDofs([dirTop, dirBot])

        matrix = mlhp.allocateSparseMatrix(mbasis1, dirichlet[0])
        vector = mlhp.allocateRhsVector(matrix)

        mintegrand = mlhp.staticDomainIntegrand(
            kinematics, degradedMaterial(pbasis1, pdofs0g1), zeroBody)
        mlhp.integrateOnDomain(
            mbasis1, mintegrand, [matrix, vector], dirichletDofs=dirichlet)

        interior = mlhp.cg(matrix, vector, rtol=1e-12,
                           M=mlhp.diagonalPreconditioner(matrix), maxiter=5000)
        mdofs1 = mlhp.inflateDofs(interior, dirichlet)

        print(f"    Mechanical: {mbasis1.nelements()} elements, "
              f"{mbasis1.ndof()} dofs", flush=True)

        # -- History update: H = max(H, psi+), materialised by local L2 ----
        updated = mlhp.meshFunctionStrainUpdate(
            history0, mbasis1, mbasis1, mdofs1, mdofs1, lib.update_address,
            kinematics=kinematics, data=[elasticParams], abi=cabi)
        history1 = mlhp.localL2Projection(grid1, updated, degree)

        # -- Phase-field Newton-Raphson ------------------------------------
        pdofs1 = pdofs0g1.copy()
        print("    || F || =", end=" ", flush=True)

        norm0 = None
        for inewton in range(maxNewton):
            pmatrix = mlhp.allocateSparseMatrix(pbasis1)
            pvector = mlhp.allocateRhsVector(pmatrix)

            pintegrand = mlhp.domainIntegrand(
                D, lib.phasefield_address,
                types=[mlhp.AssemblyType.UnsymmetricMatrix, mlhp.AssemblyType.Vector],
                maxdiff=1, fields=[history1], data=[pdofs1, pfParams], abi=cabi)
            mlhp.integrateOnDomain(pbasis1, pintegrand, [pmatrix, pvector])

            norm1 = mlhp.norm(pvector)
            norm0 = norm1 if inewton == 0 else norm0
            print(f"{norm1:.2e}", end=" ", flush=True)

            dx = mlhp.cg(pmatrix, pvector, rtol=1e-8,
                         M=mlhp.diagonalPreconditioner(pmatrix), maxiter=5000)
            pdofs1 = mlhp.add(pdofs1, dx)

            if norm1 <= max(norm0 * 1e-10, 1e-12):
                break
        print(flush=True)

        # -- Advance state -------------------------------------------------
        delta = mlhp.norm(mlhp.add(pdofs1, pdofs0g1, -1.0))
        rel = delta / max(mlhp.norm(pdofs1), 1e-30)

        pmesh0, pbasis0, pdofs0, history0 = grid1, pbasis1, pdofs1, history1

        # -- Reaction force from the mechanical residual f - Ku (no body
        #    force, so this is -Ku; at the constrained top dofs it is the
        #    reaction). Assembled on the full system (no Dirichlet). --------
        residIntegrand = mlhp.staticDomainIntegrand(
            kinematics, degradedMaterial(pbasis1, pdofs1), zeroBody,
            dofs=mdofs1, parts="vector")
        residual = mlhp.DoubleVector(mbasis1.ndof(), 0.0)
        mlhp.integrateOnDomain(mbasis1, residIntegrand, [residual])

        topX = mlhp.integrateDirichletDofs(
            mlhp.scalarField(D, 0.0), mbasis1, [3], ifield=0)[0]
        topY = mlhp.integrateDirichletDofs(
            mlhp.scalarField(D, 0.0), mbasis1, [3], ifield=1)[0]
        rarr = residual.array
        Rx, Ry = float(rarr[topX].sum()), float(rarr[topY].sum())

        print(f"    phase-field change: {rel:.2e}, reaction: "
              f"Rx={Rx:.4e} Ry={Ry:.4e}", flush=True)

        if rel < staggeredTol:
            break
    else:
        print(f"    (staggered cap {maxStaggered} reached without convergence)",
              flush=True)

    # Record the load-displacement point for every step (converged or capped).
    with open(csvPath, "a") as f:
        f.write(f"{ux};{Rx};{Ry}\n")

    # -- Postprocessing: write VTU for this load step ----------------------
    phaseField = mlhp.scalarEvaluator(pbasis0, pdofs0)
    processors = [
        mlhp.solutionProcessor(D, mdofs1, "Displacement"),
        mlhp.functionProcessor(phaseField, "PhaseField"),
        mlhp.functionProcessor(history0, "History"),
        mlhp.vonMisesProcessor(mdofs1, kinematics, degradedMaterial(pbasis0, pdofs0)),
    ]
    cellmesh = mlhp.gridCellMesh(
        mlhp.degreeOffsetResolution(mbasis1), mlhp.PostprocessTopologies.Faces)
    output = mlhp.PVtuOutput(filename=filebase + str(istep))
    mlhp.basisOutput(mbasis1, cellmesh, output, processors)

print("Done. Reaction forces in", csvPath, flush=True)

# ===========================================================================
# Quick visual check of the crack pattern with matplotlib
# ===========================================================================

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accumulator = mlhp.DataAccumulator()
    plotmesh = mlhp.gridCellMesh(
        [degree + 2] * D, mlhp.PostprocessTopologies.Faces)
    mlhp.basisOutput(
        pbasis0, plotmesh, accumulator,
        [mlhp.functionProcessor(mlhp.scalarEvaluator(pbasis0, pdofs0), "PhaseField")])

    fig, ax = plt.subplots(figsize=(6, 5))
    tri = accumulator.triangulation()
    s = np.clip(accumulator.data()[0], 0.0, 1.0)  # projection can overshoot [0,1]
    im = ax.tricontourf(tri, s, levels=np.linspace(0.0, 1.0, 21),
                        cmap="turbo", extend="both")
    ax.set_aspect("equal")
    ax.set_title(f"Phase field s (crack = blue), ux = {uxMax}")
    fig.colorbar(im, ax=ax, label="s")
    figPath = os.path.join(outputDir, "sheartest_crack.png")
    fig.savefig(figPath, dpi=150, bbox_inches="tight")
    print("Crack-pattern plot written to", figPath, flush=True)
except ImportError:
    print("matplotlib not available; skipping crack-pattern plot.", flush=True)
