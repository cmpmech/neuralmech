"""Shared building blocks for the structured-grid topology-optimization drivers.

Three reusable pieces:

* :class:`MMA` / :class:`ReferenceMMA` -- the optimizer. Single-constraint Method
  of Moving Asymptotes (Svanberg 1987): each step updates the moving asymptotes
  from the iterate history, builds the convex separable approximation, and either
  solves the one-constraint dual in closed form (``MMA``) or defers to ``mmapy``'s
  interior point (``ReferenceMMA``).
* :class:`StructuredFEM` / :class:`ComplexStructuredFEM` -- the forward solvers,
  sharing the :class:`_StructuredFEM` assembly core (dimension-agnostic grid<->element
  reshaping and the design-independent free-dof sparsity, so a SIMP loop assembles by
  gather + segmented sum into fixed CSC). Both solve with MKL pardiso
  (``solvers/mklwrapper.py``), which analyzes the sparsity once and only refactorizes
  as the design changes: ``StructuredFEM`` the real SPD system, ``ComplexStructuredFEM``
  the complex Helmholtz system ``a K + b M``. Pardiso is multithreaded, so keep OPENBLAS
  pinned to one thread to avoid oversubscription. ``CGStructuredFEM`` drops the
  factorization entirely and solves each design with Jacobi-preconditioned CG (mlhp on
  the CPU or cupyx on the GPU), warm-started from the previous design's solution.
* :class:`DensityFilter` and :func:`projection` / :func:`dprojection` -- the
  regularization: conic density filtering (with its adjoint and the classic OC
  sensitivity variant) and the smoothed-Heaviside projection pair.
"""

import math

import numpy as np
import scipy.sparse

# asymptote heuristics (Svanberg's defaults): initial span, shrink/grow factors,
# span bounds, and the small terms keeping the approximation strictly convex
ASYINIT, ASYDECR, ASYINCR = 0.5, 0.7, 1.2
ASYMIN, ASYMAX = 0.01, 10.0
RAA0, ALBEFA = 1e-5, 0.1


class MMA:
    def __init__(self, n, move=0.2):
        self.n = int(n)
        self.move = move
        self.low = np.zeros(n)
        self.upp = np.ones(n)
        self.xold1 = None
        self.xold2 = None
        self.iter = 0

    def step(self, xval, f0val, df0dx, fval, dfdx):
        # f0val is unused (the dual subsolve only needs gradients), kept so the call
        # site reads like a standard objective/constraint MMA step
        x = np.asarray(xval, dtype=float).ravel()
        df0dx = np.asarray(df0dx, dtype=float).ravel()
        dfdx = np.asarray(dfdx, dtype=float).ravel()
        fval = float(np.ravel(fval)[0])
        if self.xold1 is None:
            self.xold1 = x.copy()
            self.xold2 = x.copy()
        self.iter += 1
        low, upp = self.low, self.upp

        # move the asymptotes: widen where the design keeps moving one way, tighten
        # where it oscillates (sign of the last two steps)
        if self.iter <= 2:
            low = x - ASYINIT
            upp = x + ASYINIT
        else:
            sign = (x - self.xold1) * (self.xold1 - self.xold2)
            factor = np.where(sign > 0, ASYINCR, np.where(sign < 0, ASYDECR, 1.0))
            low = x - factor * (self.xold1 - low)
            upp = x + factor * (upp - self.xold1)
            low = np.clip(low, x - ASYMAX, x - ASYMIN)
            upp = np.clip(upp, x + ASYMIN, x + ASYMAX)

        # trust region inside the asymptotes and the move limit
        alfa = np.maximum(np.maximum(low + ALBEFA * (x - low), x - self.move), 0.0)
        beta = np.minimum(np.minimum(upp - ALBEFA * (upp - x), x + self.move), 1.0)

        # convex separable approximation coefficients (Svanberg eq. 3.2-3.4)
        ux, xl = upp - x, x - low
        p0 = (np.maximum(df0dx, 0.0) + 0.001 * np.abs(df0dx) + RAA0) * ux**2
        q0 = (np.maximum(-df0dx, 0.0) + 0.001 * np.abs(df0dx) + RAA0) * xl**2
        P = (np.maximum(dfdx, 0.0) + 0.001 * np.abs(dfdx) + RAA0) * ux**2
        Q = (np.maximum(-dfdx, 0.0) + 0.001 * np.abs(dfdx) + RAA0) * xl**2
        b = np.sum(P / ux) + np.sum(Q / xl) - fval

        # dual: x(lam) closed-form, bisect the multiplier lam >= 0 to satisfy g <= 0
        def x_of(lam):
            sp, sq = np.sqrt(p0 + lam * P), np.sqrt(q0 + lam * Q)
            return np.clip((sp * low + sq * upp) / (sp + sq), alfa, beta)

        def g(lam):
            xl_ = x_of(lam)
            return np.sum(P / (upp - xl_) + Q / (xl_ - low)) - b

        if g(0.0) <= 0.0:
            xnew = x_of(0.0)
        else:
            hi = 1.0
            while g(hi) > 0.0 and hi < 1e16:
                hi *= 2.0
            lo = 0.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if g(mid) > 0.0 else (lo, mid)
            xnew = x_of(0.5 * (lo + hi))

        self.low, self.upp = low, upp
        self.xold2, self.xold1 = self.xold1, x
        return xnew.reshape(-1, 1)


class ReferenceMMA:
    """mmapy's primal-dual interior-point MMA behind the same interface as MMA.

    Same call surface as :class:`MMA` -- ``__init__(n, move)`` and
    ``step(xval, f0val, df0dx, fval, dfdx)`` returning an ``(n, 1)`` design -- but
    defers the subproblem to ``mmapy.mmasub`` (the reference interior-point solver,
    general over the number of constraints) so the two can be swapped to cross-check
    a design. mmapy is imported lazily so it is needed only on this path.
    """

    def __init__(self, n, move=0.2):
        import mmapy

        self._mmasub = mmapy.mmasub
        self.n = int(n)
        self.move = move
        self.low = np.zeros((self.n, 1))
        self.upp = np.ones((self.n, 1))
        self.xmin = np.zeros((self.n, 1))
        self.xmax = np.ones((self.n, 1))
        self.a0 = 1.0
        self.a = np.zeros((1, 1))
        self.c = 1e3 * np.ones((1, 1))
        self.d = np.zeros((1, 1))
        self.xold1 = None
        self.xold2 = None
        self.iter = 0

    def step(self, xval, f0val, df0dx, fval, dfdx):
        xval = np.asarray(xval, dtype=float).reshape(self.n, 1)
        if self.xold1 is None:
            self.xold1 = xval.copy()
            self.xold2 = xval.copy()
        self.iter += 1
        df0dx = np.asarray(df0dx, dtype=float).reshape(self.n, 1)
        fval = np.atleast_2d(np.asarray(fval, dtype=float)).reshape(1, 1)
        dfdx = np.asarray(dfdx, dtype=float).reshape(1, self.n)
        xmma, _, _, _, _, _, _, _, _, self.low, self.upp = self._mmasub(
            1, self.n, self.iter, xval, self.xmin, self.xmax, self.xold1, self.xold2,
            f0val, df0dx, fval, dfdx, self.low, self.upp, self.a0, self.a, self.c,
            self.d, move=self.move)
        self.xold2, self.xold1 = self.xold1, xval
        return xmma


class _StructuredFEM:
    """Shared structured-grid assembly core for the topology-optimization solvers.

    Holds the dimension-agnostic grid<->element reshaping and the design-independent
    free-dof sparsity, so a SIMP loop assembles by gather + segmented sum into fixed
    CSC. Subclasses add the (real SPD or complex) factorization and solve. Physics
    enters through preintegrated unit-material element matrices ``local_mats`` (shape
    ``(n_sub, ndof_e, ndof_e)``) and grid-shaped coefficient fields the caller
    interpolates. ``grid_shape`` may be 2D or 3D.

    Args:
        efts: element location maps, shape ``(n_elems, ndof_e)``.
        free: indices of the unconstrained dofs.
        ndof: total number of dofs.
        grid_shape: design-grid resolution per axis, e.g. ``(NX, NY)``.
        sub_voxels: design sub-cells per element edge (same on every axis).
    """

    def __init__(self, efts, free, ndof, grid_shape, sub_voxels=1):
        self.efts = efts
        self.grid_shape = tuple(int(s) for s in grid_shape)
        self.sub = int(sub_voxels)
        self.dim = len(self.grid_shape)
        self.nel = [s // self.sub for s in self.grid_shape]
        self.n_elems = int(np.prod(self.nel))
        self.n_sub = self.sub**self.dim
        self.nfree = free.size

        # every element drops ndof_e^2 entries at the same (iK, jK); collapse the
        # duplicates once so each assembly is a gather + segmented sum into fixed CSC
        # (equivalent to scipy.sparse.coo_matrix(...).tocsc() per design, but ~2x faster)
        ndof_e = efts.shape[1]
        iK = np.repeat(efts, ndof_e, axis=1).ravel()
        jK = np.tile(efts, (1, ndof_e)).ravel()
        dof_map = np.full(ndof, -1)
        dof_map[free] = np.arange(free.size)
        keep = (dof_map[iK] >= 0) & (dof_map[jK] >= 0)  # both dofs free
        ri, rj = dof_map[iK[keep]], dof_map[jK[keep]]
        order = np.lexsort((ri, rj))  # column-major order expected by CSC
        self._data_idx = np.flatnonzero(keep)[order]  # gather into mat_e.ravel()
        ri, rj = ri[order], rj[order]
        first = np.empty(ri.size, dtype=bool)
        first[0] = True
        first[1:] = (ri[1:] != ri[:-1]) | (rj[1:] != rj[:-1])
        self._seg = np.flatnonzero(first)  # duplicate (row, col) group boundaries
        self._indices = ri[first].astype(np.int32)
        self._indptr = np.concatenate(
            [[0], np.cumsum(np.bincount(rj[first], minlength=free.size))]
        ).astype(np.int32)

        # data positions of the diagonal, for optional nodal springs (compliant
        # mechanisms): every free dof self-couples, so each diagonal entry exists
        cols = np.repeat(np.arange(self.nfree), np.diff(self._indptr))
        self._diag_idx = np.flatnonzero(self._indices == cols)

    def grid_to_elements(self, field):  # grid -> (n_elems, n_sub)
        interleaved = np.empty(2 * self.dim, dtype=int)
        interleaved[0::2] = self.nel
        interleaved[1::2] = self.sub
        perm = list(range(0, 2 * self.dim, 2)) + list(range(1, 2 * self.dim, 2))
        return (
            field.reshape(interleaved).transpose(perm).reshape(self.n_elems, self.n_sub)
        )

    def elements_to_grid(self, field):  # (n_elems, n_sub) -> grid
        shape = self.nel + [self.sub] * self.dim
        perm = [ax for d in range(self.dim) for ax in (d, self.dim + d)]
        return field.reshape(shape).transpose(perm).reshape(self.grid_shape)

    def assemble(self, coeff, local_mats):  # per-voxel-scaled local_mats -> free-free CSC
        scale_e = self.grid_to_elements(coeff)
        mat_e = np.einsum("es,sij->eij", scale_e, local_mats, optimize=True)
        data = np.add.reduceat(mat_e.ravel()[self._data_idx], self._seg)
        return scipy.sparse.csc_matrix(
            (data, self._indices, self._indptr), shape=(self.nfree, self.nfree)
        )

    def bilinear(self, local_mats, left, right):  # grid-shaped left^T local_mats right
        l_e, r_e = left[self.efts], right[self.efts]
        # one gemm beats np.einsum's 3-operand path ~3x: tmp[e,s,i] = K[s,i,j] r[e,j]
        flat = np.asarray(local_mats).transpose(2, 0, 1).reshape(r_e.shape[1], -1)
        tmp = (r_e @ flat).reshape(len(r_e), -1, l_e.shape[1])
        return self.elements_to_grid(np.einsum("esi,ei->es", tmp, l_e))


class StructuredFEM(_StructuredFEM):
    """Penalized real-SPD FEM assemble-and-solve on a structured grid, free dofs only.

    Backed by MKL pardiso: it registers its analysis on the first assembled system
    and only refactorizes as the design changes, so each SIMP step refreshes just the
    matrix values. ``K_locals`` are the unit-material element matrices, so the same
    object serves scalar (heat) and vector (elasticity) problems. Pardiso runs on MKL
    threads, so drivers should pin OPENBLAS to one thread. See :class:`_StructuredFEM`
    for the shared arguments.
    """

    def __init__(self, efts, free, ndof, K_locals, grid_shape, sub_voxels=1):
        super().__init__(efts, free, ndof, grid_shape, sub_voxels)
        from solvers.mklwrapper import pardisoFactorize

        self._pardiso = pardisoFactorize
        self._factor = None
        self.K_locals = K_locals

    def solve(self, material, rhs_free, spring_diag=None):  # K(material) u = rhs, free dofs
        K = self.assemble(material, self.K_locals)
        if spring_diag is not None:  # add nodal springs onto the diagonal (mechanisms)
            K.data[self._diag_idx] += spring_diag
        # K is symmetric, so its CSC transpose is a free CSR view of the same system
        if self._factor is None:
            self._factor = self._pardiso(K.T, symmetric=True)
        else:
            self._factor.refactorize(K.T)
        return self._factor(np.asarray(rhs_free, dtype=float))

    def element_energy(self, u):  # grid-shaped unit-material energy u^T K_local u
        return self.bilinear(self.K_locals, u, u)


class CGStructuredFEM(_StructuredFEM):
    """Drop-in :class:`StructuredFEM` variant solved with Jacobi-preconditioned CG.

    Matrix-explicit but factorization-free: each design is assembled into the fixed
    CSC and solved iteratively, warm-started from the previous design's solution.
    ``use_cupy=False`` runs mlhp's CG with the scipy matvec wrapped as a linear
    operator; ``use_cupy=True`` keeps one CSR copy on the GPU (refreshing only the
    values, since the sparsity is design-independent) and solves with cupyx CG.
    Iteration counts grow with the SIMP contrast ``E0/EMIN``, so the direct solvers
    stay faster on small 2D problems. Single right-hand side only. See
    :class:`StructuredFEM` for the shared arguments.

    Args:
        use_cupy: solve on the GPU with cupyx CG instead of mlhp CG on the CPU.
        rtol: relative residual tolerance of the CG solve.
        maxiter: CG iteration cap per solve.
    """

    def __init__(self, efts, free, ndof, K_locals, grid_shape, sub_voxels=1,
                 use_cupy=False, rtol=1e-8, maxiter=10000):
        super().__init__(efts, free, ndof, grid_shape, sub_voxels)
        self.K_locals = K_locals
        self.use_cupy = use_cupy
        self.rtol = rtol
        self.maxiter = maxiter
        self._x0 = np.zeros(self.nfree)
        if use_cupy:
            import cupy
            import cupyx.scipy.sparse
            import cupyx.scipy.sparse.linalg

            self._cp = cupy
            self._cpx_sparse = cupyx.scipy.sparse
            self._cpx_linalg = cupyx.scipy.sparse.linalg
            self._K_gpu = None
        else:
            import mlhp

            self._mlhp = mlhp

    def solve(self, material, rhs_free, spring_diag=None):  # K(material) u = rhs, free dofs
        K = self.assemble(material, self.K_locals)
        if spring_diag is not None:  # add nodal springs onto the diagonal (mechanisms)
            K.data[self._diag_idx] += spring_diag
        diag = K.diagonal()
        rhs = np.asarray(rhs_free, dtype=float)
        if self.use_cupy:
            cp = self._cp
            # K is symmetric, so its CSC arrays double as a CSR view of the same system
            if self._K_gpu is None:
                self._K_gpu = self._cpx_sparse.csr_matrix(
                    (cp.asarray(K.data), cp.asarray(K.indices), cp.asarray(K.indptr)),
                    shape=K.shape)
            else:
                self._K_gpu.data[:] = cp.asarray(K.data)
            inv_diag = 1.0 / cp.asarray(diag)
            M = self._cpx_linalg.LinearOperator(K.shape, matvec=lambda x: inv_diag * x)
            u, info = self._cpx_linalg.cg(
                self._K_gpu, cp.asarray(rhs), x0=cp.asarray(self._x0),
                rtol=self.rtol, maxiter=self.maxiter, M=M)
            u = cp.asnumpy(u)
        else:
            mlhp = self._mlhp
            A = mlhp.linearOperator_array(lambda x, out: np.copyto(out, K @ x))
            M = mlhp.linearOperator_array(lambda x, out: np.divide(x, diag, out=out))
            u = mlhp.cg(
                A, mlhp.DoubleVector(rhs.tolist()), x0=mlhp.DoubleVector(self._x0.tolist()),
                rtol=self.rtol, maxiter=self.maxiter, M=M)
            u = np.array(u.array)
        self._x0 = u
        return u

    def element_energy(self, u):  # grid-shaped unit-material energy u^T K_local u
        return self.bilinear(self.K_locals, u, u)


class ComplexStructuredFEM(_StructuredFEM):
    """Complex Helmholtz assemble-and-solve on a structured grid, backed by MKL pardiso.

    The system ``S = a K + b M`` mixes a stiffness (K) and a mass (M) element matrix,
    each scaled by its own per-voxel coefficient field, with ``b`` allowed complex
    (e.g. a damped ``-(i omega eta + omega^2) kappa^-1``). Its sparsity is
    design-independent, so pardiso analyzes it once and only refactorizes per design
    (complex structurally symmetric). The constant ``dS/dvar`` operator is left to the
    caller, who builds it from ``K_locals``/``M_locals`` and feeds it to
    :meth:`bilinear` for the adjoint sensitivity. See :class:`_StructuredFEM` for the
    shared arguments.
    """

    def __init__(self, efts, free, ndof, K_locals, M_locals, grid_shape, sub_voxels=1):
        super().__init__(efts, free, ndof, grid_shape, sub_voxels)
        from solvers.mklwrapper import pardisoFactorize

        self._pardiso = pardisoFactorize
        self._factor = None
        self.K_locals = K_locals
        self.M_locals = M_locals

    def system(self, coeff_k, coeff_m):  # a(x) K + b(x) M as a complex CSC
        return (
            self.assemble(coeff_k, self.K_locals) + self.assemble(coeff_m, self.M_locals)
        ).tocsc()

    def factorize(self, matrix):  # pardiso refactorization on the fixed sparsity
        # S is symmetric, so its CSC transpose is a free CSR view of the same system
        if self._factor is None:
            self._factor = self._pardiso(matrix.T, symmetric=True)
        else:
            self._factor.refactorize(matrix.T)
        return self._factor


class DensityFilter:
    """Conic density filter on a structured 2D grid, with adjoint and OC variants.

    Precomputes the radius-``rmin`` conic kernel and its normalization on a grid of
    ``shape``. Works on NumPy arrays by default; pass ``xp=cupy`` (and optionally a
    ``dtype``) to build and convolve on the GPU instead.
    """

    def __init__(self, rmin, shape, xp=np, dtype=None):
        if xp is np:
            import scipy.ndimage as ndi
        else:
            import cupyx.scipy.ndimage as ndi
        self.xp, self.ndi = xp, ndi
        ceil_r = int(math.ceil(rmin))
        ki, kj = xp.meshgrid(
            xp.arange(-ceil_r, ceil_r + 1),
            xp.arange(-ceil_r, ceil_r + 1),
            indexing="ij",
        )
        self.kernel = xp.maximum(0.0, rmin - xp.sqrt(ki**2 + kj**2))
        if dtype is not None:
            self.kernel = self.kernel.astype(dtype)
        self.Hs = ndi.convolve(
            xp.ones(shape, self.kernel.dtype), self.kernel, mode="constant", cval=0.0
        )

    def __call__(self, x):  # conic smoothing of the raw design
        return self.ndi.convolve(x, self.kernel, mode="constant", cval=0.0) / self.Hs

    def adjoint(self, g):  # transpose of the filter, for the chain rule
        return self.ndi.convolve(g / self.Hs, self.kernel, mode="constant", cval=0.0)

    def sensitivity(self, rho, dc):  # classic OC sensitivity filter (Sigmund 2001)
        num = self.ndi.convolve(rho * dc, self.kernel, mode="constant", cval=0.0)
        return num / (self.xp.maximum(rho, 1e-3) * self.Hs)


def projection(x, beta, eta):
    """Smoothed Heaviside about threshold ``eta`` (NumPy or CuPy array ``x``)."""
    a, b = math.tanh(beta * eta), math.tanh(beta * (1.0 - eta))
    return (a + np.tanh(beta * (x - eta))) / (a + b)


def dprojection(x, beta, eta):
    """Derivative of :func:`projection` with respect to ``x``."""
    a, b = math.tanh(beta * eta), math.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (x - eta)) ** 2) / (a + b)
