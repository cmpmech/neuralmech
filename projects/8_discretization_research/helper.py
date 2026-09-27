import math
import time

import mlhp
import numpy as np
import scipy.signal

from solvers.mklwrapper import pardisoFactorize
from solvers.optimization import (
    MMA,
    DensityFilter,
    StructuredFEM,
    _StructuredFEM,
    dprojection,
    dsimp,
    projection,
    simp,
)

LENGTHS = [2.0, 1.0]
E0, EMIN, NU = 1.0, 1e-9, 0.3
PENAL = 3.0
TRACTION = 1.0
LOAD_SPAN = (0.4, 0.6)  # traction patch on the right edge; element-aligned for S | 24
MBB_LENGTHS = [3.0, 1.0]
MBB_PATCH = 16  # voxel width of the load and roller patches of the MBB beam


class Analysis:
    """voxel-grid cantilever analyzed with S x S subvoxel elements of degree p.

    The voxel moduli are integrated exactly in each element (finite cell style), so the
    element stiffness is the modulus-weighted sum of the S^2 preintegrated subvoxel
    matrices. `compliance(rho)` returns (c, dc/drho) on the voxel grid. Subclasses swap
    the problem through `lengths`, `boundary` and `passive`.
    """

    lengths = LENGTHS

    def __init__(self, nvoxels, sub, degree):
        assert all(n % sub == 0 for n in nvoxels)
        self.nvoxels, self.sub = tuple(nvoxels), sub
        self.nel = [n // sub for n in nvoxels]
        elem_lengths = [length / n for length, n in zip(self.lengths, self.nel)]

        mesh = mlhp.makeRefinedGrid(
            mlhp.makeGrid(ncells=self.nel, lengths=self.lengths)
        )
        basis = mlhp.makeHpTensorSpace(mesh, degree=degree, nfields=2)
        self.ndof = basis.ndof()
        efts = np.array(basis.locationMaps())

        mesh_local = mlhp.makeRefinedGrid(
            mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths)
        )
        basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=degree, nfields=2)
        material = mlhp.planeStressMaterial(
            mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU)
        )
        integrand = mlhp.staticDomainIntegrand(
            mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
        )
        K_locals = mlhp.integratePartitionMatrices(
            basis_local,
            integrand,
            mlhp.gridQuadrature(nsubcells=[sub, sub]),
            mlhp.absoluteQuadratureOrder([degree + 1, degree + 1]),
        )

        dirichlet, face, traction = self.boundary(basis, efts)
        self.free = np.setdiff1d(np.arange(self.ndof), dirichlet[0])
        neumann = mlhp.neumannIntegrand(mlhp.vectorField(2, traction))
        vector = mlhp.allocateRhsVector(mlhp.allocateSparseMatrix(basis, dirichlet[0]))
        mlhp.integrateOnSurface(
            basis,
            neumann,
            [vector],
            mlhp.quadratureOnMeshFaces(mesh, [face]),
            dirichletDofs=dirichlet,
        )
        self.force_free = np.array(vector.array)
        self.fem = StructuredFEM(
            efts, self.free, self.ndof, K_locals, self.nvoxels, sub
        )

    def boundary(self, basis, efts):  # (dirichlet dofs, loaded face, traction)
        zero = mlhp.scalarField(2, 0.0)
        clamp = mlhp.combineDirichletDofs(
            [mlhp.integrateDirichletDofs(zero, basis, [0], ifield=i) for i in range(2)]
        )
        lo, hi = LOAD_SPAN
        return clamp, 1, f"[0.0, -{TRACTION} if y > {lo} and y < {hi} else 0.0]"

    @staticmethod
    def passive(nvoxels):
        return load_patch(nvoxels)

    def compliance(self, rho):
        u = np.zeros(self.ndof)
        u[self.free] = self.fem.solve(simp(rho, PENAL, EMIN, E0), self.force_free)
        c = self.force_free @ u[self.free]
        dc = -dsimp(rho, PENAL, EMIN, E0) * self.fem.element_energy(u)
        return c, dc


class MBBAnalysis(Analysis):
    """half MBB beam (left edge is the symmetry plane) of `Analysis`.

    Load and roller act on patches `MBB_PATCH` voxels wide instead of points, since a
    point load or support makes the voxel-level compliance diverge.
    """

    lengths = MBB_LENGTHS

    def boundary(self, basis, efts):
        assert MBB_PATCH % self.sub == 0
        zero = mlhp.scalarField(2, 0.0)
        symmetry = mlhp.integrateDirichletDofs(zero, basis, [0], ifield=0)

        # vertical dofs of the bottom edges of the patch elements, exact for any degree
        bottom = mlhp.integrateDirichletDofs(zero, basis, [2], ifield=1)[0]
        nelx, nely = self.nel
        patch = [ex * nely for ex in range(nelx - MBB_PATCH // self.sub, nelx)]
        roller = np.intersect1d(bottom, efts[patch]).tolist()
        dirichlet = mlhp.combineDirichletDofs([symmetry, (roller, [0.0] * len(roller))])

        width = MBB_PATCH * self.lengths[1] / self.nvoxels[1]
        return dirichlet, 3, f"[0.0, -{TRACTION} if x < {width} else 0.0]"

    @staticmethod
    def passive(nvoxels, depth=2):
        solid = np.zeros(nvoxels, dtype=bool)
        solid[:MBB_PATCH, -depth:] = True
        solid[-MBB_PATCH:, :depth] = True
        return solid


def voxel_stiffness():
    """unit-modulus plane-stress Q1 voxel matrix, nodes (0,0), (0,1), (1,0), (1,1)."""
    D = np.array([[1.0, NU, 0.0], [NU, 1.0, 0.0], [0.0, 0.0, (1.0 - NU) / 2]])
    D /= 1.0 - NU**2
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    gauss = [0.5 - 0.5 / math.sqrt(3), 0.5 + 0.5 / math.sqrt(3)]
    k = np.zeros((8, 8))
    for gx in gauss:
        for gy in gauss:
            wx = np.where(corners[:, 0] == 1, gx, 1 - gx)
            wy = np.where(corners[:, 1] == 1, gy, 1 - gy)
            dNx = np.where(corners[:, 0] == 1, 1.0, -1.0) * wy
            dNy = np.where(corners[:, 1] == 1, 1.0, -1.0) * wx
            B = np.zeros((3, 8))
            B[0, 0::2], B[1, 1::2] = dNx, dNy
            B[2, 0::2], B[2, 1::2] = dNy, dNx
            k += 0.25 * B.T @ D @ B
    return k


class MultiscaleAnalysis:
    """cantilever of `Analysis` with multiscale elements instead of polynomial ones.

    Each element carries degree-p Lagrange traces on its edges only; its interior is the
    voxel-Q1 field, condensed per element on every analysis. The coarse space is thus a
    subspace of the voxel-Q1 space that resolves how voxels inside an element connect.
    """

    def __init__(self, nvoxels, sub, degree):
        S, p = sub, degree
        self.nvoxels, self.sub = tuple(nvoxels), sub
        self.nel = (nvoxels[0] // S, nvoxels[1] // S)
        nelx, nely = self.nel

        # fine element nodes (i, j) -> i * (S + 1) + j, dofs 2 * node + component
        nf = (S + 1) ** 2

        def node(i, j):
            return i * (S + 1) + j

        self.k = voxel_stiffness()
        self.vox_dofs = np.array(
            [
                [2 * n + c for n in corners for c in range(2)]
                for a in range(S)
                for b in range(S)
                for corners in [
                    [node(a, b), node(a, b + 1), node(a + 1, b), node(a + 1, b + 1)]
                ]
            ]
        )
        T = np.zeros((S * S, 2 * nf, 2 * nf))
        for s, dofs in enumerate(self.vox_dofs):
            T[s][np.ix_(dofs, dofs)] = self.k
        self.T = T.reshape(S * S, -1)

        ii, jj = np.meshgrid(np.arange(S + 1), np.arange(S + 1), indexing="ij")
        on_edge = np.repeat(((ii == 0) | (ii == S) | (jj == 0) | (jj == S)).ravel(), 2)
        self.bdofs, self.idofs = np.flatnonzero(on_edge), np.flatnonzero(~on_edge)

        # Lagrange traces at the fine nodes; bubble nodes drop out (zero on the edges)
        t = np.arange(p + 1) / p
        L = np.array(
            [
                [
                    np.prod(
                        [(x - t[m]) / (t[k] - t[m]) for m in range(p + 1) if m != k]
                    )
                    for k in range(p + 1)
                ]
                for x in np.arange(S + 1) / S
            ]
        )
        coarse = [(a, b) for a in range(p + 1) for b in range(p + 1) if {a, b} & {0, p}]
        N = np.array([np.outer(L[:, a], L[:, b]).ravel() for a, b in coarse]).T
        self.B = np.zeros((2 * nf, 2 * len(coarse)))
        self.B[0::2, 0::2], self.B[1::2, 1::2] = N, N

        # global numbering of the coarse nodes on element edges
        gx, gy = p * nelx + 1, p * nely + 1
        gi, gj = np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij")
        used = ((gi % p == 0) | (gj % p == 0)).ravel()
        node_id = np.full(gx * gy, -1)
        node_id[used] = np.arange(used.sum())
        self.efts = np.array(
            [
                [
                    2 * node_id[(ex * p + a) * gy + ey * p + b] + c
                    for a, b in coarse
                    for c in range(2)
                ]
                for ex in range(nelx)
                for ey in range(nely)
            ]
        )
        self.ndof = 2 * int(used.sum())
        fixed = np.flatnonzero(np.repeat(gi.ravel()[used] == 0, 2))
        self.free = np.setdiff1d(np.arange(self.ndof), fixed)

        # fine nodal forces of the traction patch, restricted by the traces
        h = LENGTHS[1] / nvoxels[1]
        f_e = np.zeros((nelx, nely, 2 * nf))
        for ey in range(nely):
            for j in range(S):
                y0 = (ey * S + j) * h
                if LOAD_SPAN[0] - 1e-12 <= y0 and y0 + h <= LOAD_SPAN[1] + 1e-12:
                    f_e[-1, ey, 2 * node(S, j) + 1] -= TRACTION * h / 2
                    f_e[-1, ey, 2 * node(S, j + 1) + 1] -= TRACTION * h / 2
        force = np.zeros(self.ndof)
        np.add.at(force, self.efts, f_e.reshape(-1, 2 * nf) @ self.B)
        self.force_free = force[self.free]

        self.fem = _StructuredFEM(self.efts, self.free, self.ndof, self.nel, upper=True)
        self._factor = None

    def to_elements(self, field):
        (nx, ny), S = self.nel, self.sub
        return field.reshape(nx, S, ny, S).transpose(0, 2, 1, 3).reshape(nx * ny, -1)

    def to_grid(self, field):
        (nx, ny), S = self.nel, self.sub
        return field.reshape(nx, ny, S, S).transpose(0, 2, 1, 3).reshape(nx * S, -1)

    def element_matrices(self, E):
        nf2 = self.B.shape[0]
        Kf = (self.to_elements(E) @ self.T).reshape(-1, nf2, nf2)
        bd, idd = self.bdofs, self.idofs
        Phi = np.empty((len(Kf), nf2, self.B.shape[1]))
        Phi[:, bd] = self.B[bd]
        Phi[:, idd] = -np.linalg.solve(
            Kf[:, idd][:, :, idd], Kf[:, idd][:, :, bd] @ self.B[bd]
        )
        Kc = Phi.transpose(0, 2, 1) @ (Kf @ Phi)
        return 0.5 * (Kc + Kc.transpose(0, 2, 1)), Phi

    def compliance(self, rho):
        Kc, Phi = self.element_matrices(simp(rho, PENAL, EMIN, E0))
        K = self.fem.matrix(self.fem._scatter @ Kc.ravel())
        if self._factor is None:
            self._factor = pardisoFactorize(K, symmetric=True, symmetricHalf=True)
        else:
            self._factor.refactorize(K)
        u = np.zeros(self.ndof)
        u[self.free] = self._factor(self.force_free)
        c = self.force_free @ u[self.free]

        # the condensed interior is energy-optimal: dK_e/dE_s sees just the fine field
        u_fine = (Phi @ u[self.efts][:, :, None])[:, :, 0]
        u_vox = u_fine[:, self.vox_dofs]
        energy = ((u_vox @ self.k) * u_vox).sum(-1)
        return c, -dsimp(rho, PENAL, EMIN, E0) * self.to_grid(energy)


class FilterProject:
    """`rmin`-voxel density filter, projected about `eta` once beta is set."""

    def __init__(self, rmin, shape, eta=0.5):
        self.filter, self.eta = DensityFilter(rmin, shape), eta

    def __call__(self, x, beta):
        self.x_tilde, self.beta = self.filter(x), beta
        if beta is None:
            return self.x_tilde
        return projection(self.x_tilde, beta, self.eta)

    def adjoint(self, g):
        if self.beta is not None:
            g = g * dprojection(self.x_tilde, self.beta, self.eta)
        return self.filter.adjoint(g)


class Project:
    """Heaviside projection about `eta`, the identity until beta is set."""

    def __init__(self, eta=0.5):
        self.eta = eta

    def __call__(self, x, beta):
        self.x, self.beta = x, beta
        return x if beta is None else projection(x, beta, self.eta)

    def adjoint(self, g):
        if self.beta is None:
            return g
        return g * dprojection(self.x, self.beta, self.eta)


class Solid:
    """passive solid voxels, placed before the closing so adjacent gaps close too."""

    def __init__(self, mask):
        self.mask = mask

    def __call__(self, x, beta):
        return np.where(self.mask, 1.0, x)

    def adjoint(self, g):
        return np.where(self.mask, 0.0, g)


class Dilate:
    """smooth dilation over a flat disc of radius R voxels (log-sum-exp mean).

    Tends to the exact maximum as beta grows. The domain is mirrored at its boundary, so
    the outside counts as neither solid nor void. The FFT convolution makes the cost
    independent of R but needs exp(beta) far below 1e16, hence the moderate beta.

    Reference: https://doi.org/10.1007/s00158-006-0087-x
    """

    def __init__(self, radius, shape, beta=20.0):
        # +0.5 blunts the one-pixel disc tips that slip into gaps across thin struts
        m = math.ceil(radius + 0.5)
        r = np.arange(-m, m + 1)
        disc = r[:, None] ** 2 + r[None, :] ** 2 <= (radius + 0.5) ** 2
        self.kernel = disc.astype(float)
        self.beta = beta
        self.pad_ids = np.pad(np.arange(np.prod(shape)).reshape(shape), m, "symmetric")

    def __call__(self, x, beta=None):
        self.ex = np.exp(self.beta * x)
        self.A = scipy.signal.fftconvolve(
            self.ex.ravel()[self.pad_ids], self.kernel, "valid"
        )
        return np.log(self.A / self.kernel.sum()) / self.beta

    def adjoint(self, g):
        full = scipy.signal.fftconvolve(g / self.A, self.kernel, "full")
        folded = np.bincount(self.pad_ids.ravel(), full.ravel(), g.size)
        return self.ex * folded.reshape(g.shape)


class Erode(Dilate):
    """smooth erosion, the dilation of the void."""

    def __call__(self, x, beta=None):
        return 1.0 - super().__call__(1.0 - x)


class DesignMap:
    """chain of design operators; `beta` drives every projection in the chain."""

    def __init__(self, ops):
        self.ops, self.beta = ops, None

    def __call__(self, x):
        for op in self.ops:
            x = op(x, self.beta)
        return x

    def adjoint(self, g):
        for op in self.ops[::-1]:
            g = op.adjoint(g)
        return g


def classic_map(shape, solid, rmin=1.5):
    """density filter + projection, the solid and void length scale are both rmin."""
    return DesignMap([FilterProject(rmin, shape), Solid(solid)])


def closing_map(shape, solid, radius, rmin=1.5):
    """classic map, then a closing that fills every void narrower than about 2 radius.

    The closing only ever adds material, so the solid length scale stays rmin.
    """
    return DesignMap(
        [
            FilterProject(rmin, shape),
            Solid(solid),
            Dilate(radius, shape),
            Erode(radius, shape),
            Project(),
        ]
    )


def load_patch(nvoxels, depth=2):
    """voxels under the traction, kept solid so the load never acts on void."""
    solid = np.zeros(nvoxels, dtype=bool)
    lo, hi = (round(y * nvoxels[1] / LENGTHS[1]) for y in LOAD_SPAN)
    solid[-depth:, lo:hi] = True
    return solid


def optimize(analysis, design_map, volfrac, iters, betas, move=0.1):
    """MMA compliance minimization under a volume constraint.

    `betas` maps iteration -> projection sharpness (continuation). Returns the physical
    design, the compliance history and the seconds spent in (design map, analysis).
    """
    x = np.full(analysis.nvoxels, volfrac)
    mma = MMA(x.size, move)
    history, timings = [], np.zeros(2)
    c0 = None
    for it in range(iters):
        design_map.beta = betas.get(it, design_map.beta)
        tic = time.perf_counter()
        rho = design_map(x)
        tac = time.perf_counter()
        c, dc = analysis.compliance(rho)
        toc = time.perf_counter()
        dc_x = design_map.adjoint(dc)
        dv_x = design_map.adjoint(np.full(rho.shape, 1.0 / (rho.size * volfrac)))
        timings += [time.perf_counter() - toc + tac - tic, toc - tac]
        c0 = c if c0 is None else c0
        x = mma.step(
            x.ravel(),
            c / c0,
            dc_x.ravel() / c0,
            rho.mean() / volfrac - 1.0,
            dv_x.ravel(),
        )
        x = x.reshape(rho.shape)
        history.append(c)
    return design_map(x), np.array(history), timings
