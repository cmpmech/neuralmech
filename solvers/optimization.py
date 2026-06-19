"""Shared building blocks for the structured-grid topology-optimization drivers.

Two reusable pieces:

* :class:`MMA` / :class:`ReferenceMMA` -- the optimizer. Single-constraint Method
  of Moving Asymptotes (Svanberg 1987): each step updates the moving asymptotes
  from the iterate history, builds the convex separable approximation, and either
  solves the one-constraint dual in closed form (``MMA``) or defers to ``mmapy``'s
  interior point (``ReferenceMMA``).
* :class:`StructuredFEM` / :class:`ComplexStructuredFEM` -- the forward solvers,
  sharing the :class:`_StructuredFEM` assembly core (dimension-agnostic grid<->element
  reshaping and the design-independent free-dof sparsity, so a SIMP loop assembles by
  gather + segmented sum into fixed CSC). ``StructuredFEM`` adds a real SPD solve with
  one reused CHOLMOD factorization; ``ComplexStructuredFEM`` adds a complex Helmholtz
  system ``a K + b M`` solved with a fresh sparse LU each design.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

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
        return self.elements_to_grid(
            np.einsum("ei,sij,ej->es", l_e, local_mats, r_e, optimize=True)
        )


class StructuredFEM(_StructuredFEM):
    """Penalized real-SPD FEM assemble-and-solve on a structured grid, free dofs only.

    One symbolic CHOLMOD factorization is reused across the SIMP loop; each call
    refreshes only the matrix values as the design changes. ``K_locals`` are the
    unit-material element matrices, so the same object serves scalar (heat) and
    vector (elasticity) problems. See :class:`_StructuredFEM` for the shared arguments.
    """

    def __init__(self, efts, free, ndof, K_locals, grid_shape, sub_voxels=1):
        super().__init__(efts, free, ndof, grid_shape, sub_voxels)
        import cvxopt
        import cvxopt.cholmod

        self._cvxopt = cvxopt
        self._cholmod = cvxopt.cholmod
        self.K_locals = K_locals

        # the sparsity is design-independent, so factor it symbolically once
        K_free = self.assemble(np.ones(self.grid_shape), K_locals)
        self._A = cvxopt.spmatrix(
            cvxopt.matrix(K_free.data),
            cvxopt.matrix(K_free.indices.tolist()),
            cvxopt.matrix(np.repeat(np.arange(self.nfree), np.diff(K_free.indptr)).tolist()),
            (self.nfree, self.nfree),
        )
        self._factor = cvxopt.cholmod.symbolic(self._A)

    def solve(self, material, rhs_free):  # solve K(material) u = rhs over free dofs
        self._A.V = self._cvxopt.matrix(self.assemble(material, self.K_locals).data)
        self._cholmod.numeric(self._A, self._factor)
        b = self._cvxopt.matrix(np.asarray(rhs_free, dtype=float))
        self._cholmod.solve(self._factor, b)
        return np.array(b).ravel()

    def element_energy(self, u):  # grid-shaped unit-material energy u^T K_local u
        return self.bilinear(self.K_locals, u, u)


class ComplexStructuredFEM(_StructuredFEM):
    """Complex Helmholtz assemble-and-solve on a structured grid.

    The system ``S = a K + b M`` mixes a stiffness (K) and a mass (M) element matrix,
    each scaled by its own per-voxel coefficient field, with ``b`` allowed complex
    (e.g. a damped ``-(i omega eta + omega^2) kappa^-1``). S is complex and indefinite,
    so it is refactored with a fresh sparse LU each design -- there is no symbolic
    reuse to exploit as in the real SPD case. The constant ``dS/dvar`` operator is left
    to the caller, who builds it from ``K_locals``/``M_locals`` and feeds it to
    :meth:`bilinear` for the adjoint sensitivity. See :class:`_StructuredFEM` for the
    shared arguments.
    """

    def __init__(self, efts, free, ndof, K_locals, M_locals, grid_shape, sub_voxels=1):
        super().__init__(efts, free, ndof, grid_shape, sub_voxels)
        self.K_locals = K_locals
        self.M_locals = M_locals

    def system(self, coeff_k, coeff_m):  # a(x) K + b(x) M as a complex CSC
        return (
            self.assemble(coeff_k, self.K_locals) + self.assemble(coeff_m, self.M_locals)
        ).tocsc()

    def factorize(self, matrix):  # complex sparse LU (no symbolic reuse across designs)
        return scipy.sparse.linalg.splu(matrix)
