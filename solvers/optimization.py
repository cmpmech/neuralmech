"""MMA, structured-grid FEM, and filters shared by the topology optimization drivers."""

import math

import numpy as np
import scipy.optimize
import scipy.sparse

# asymptote heuristics (Svanberg's defaults): initial span, shrink/grow factors,
# span bounds, and the small terms keeping the approximation strictly convex
ASYINIT, ASYDECR, ASYINCR = 0.5, 0.7, 1.2
ASYMIN, ASYMAX = 0.01, 10.0
RAA0, ALBEFA = 1e-5, 0.1


class MMA:
    """method of moving asymptotes for one constraint, with a dual root-find subsolve.

    `step(xval, f0val, df0dx, fval, dfdx)` returns the next (n, 1) design in [0, 1].

    Reference: https://doi.org/10.1002/nme.1620240207
    """

    def __init__(self, n, move=0.2):
        self.move = move
        self.low = self.upp = self.xold1 = self.xold2 = None
        self.iter = 0

    def step(self, xval, f0val, df0dx, fval, dfdx):
        # f0val is unused (the dual subsolve only needs gradients), kept so the call
        # site reads like a standard objective/constraint MMA step
        x = np.asarray(xval, dtype=float).ravel()
        fval = float(np.ravel(fval)[0])
        if self.xold1 is None:
            self.xold1 = self.xold2 = x
        self.iter += 1

        # move the asymptotes: widen where the design keeps moving one way, tighten
        # where it oscillates (sign of the last two steps)
        if self.iter <= 2:
            low, upp = x - ASYINIT, x + ASYINIT
        else:
            sign = (x - self.xold1) * (self.xold1 - self.xold2)
            factor = np.where(sign > 0, ASYINCR, np.where(sign < 0, ASYDECR, 1.0))
            low = np.clip(x - factor * (self.xold1 - self.low), x - ASYMAX, x - ASYMIN)
            upp = np.clip(x + factor * (self.upp - self.xold1), x + ASYMIN, x + ASYMAX)

        # trust region inside the asymptotes and the move limit
        alfa = np.maximum(np.maximum(low + ALBEFA * (x - low), x - self.move), 0.0)
        beta = np.minimum(np.minimum(upp - ALBEFA * (upp - x), x + self.move), 1.0)

        # convex separable approximation coefficients (Svanberg eq. 3.2-3.4)
        ux2, xl2 = (upp - x) ** 2, (x - low) ** 2

        def pq(df):
            df = np.asarray(df, dtype=float).ravel()
            eps = 0.001 * np.abs(df) + RAA0
            return (np.maximum(df, 0.0) + eps) * ux2, (np.maximum(-df, 0.0) + eps) * xl2

        (p0, q0), (P, Q) = pq(df0dx), pq(dfdx)
        b = np.sum(P / (upp - x)) + np.sum(Q / (x - low)) - fval

        # dual: x(lam) is closed-form, root-find the multiplier lam >= 0 of g <= 0
        # (g decreases monotonically in lam)
        def x_of(lam):
            sp, sq = np.sqrt(p0 + lam * P), np.sqrt(q0 + lam * Q)
            return np.clip((sp * low + sq * upp) / (sp + sq), alfa, beta)

        def g(lam):
            x_lam = x_of(lam)
            return np.sum(P / (upp - x_lam) + Q / (x_lam - low)) - b

        lam = 0.0
        if g(0.0) > 0.0:
            hi = 1.0
            while (g_hi := g(hi)) > 0.0 and hi < 1e16:
                hi *= 2.0
            lam = hi if g_hi > 0.0 else scipy.optimize.brentq(g, 0.0, hi, rtol=1e-15)

        self.low, self.upp = low, upp
        self.xold2, self.xold1 = self.xold1, x
        return x_of(lam).reshape(-1, 1)


class ReferenceMMA:
    """mmapy's primal-dual interior-point MMA behind the interface of `MMA`.

    Defers the subproblem to `mmapy.mmasub` so the two can be swapped to cross-check a
    design; mmapy is imported lazily.
    """

    def __init__(self, n, move=0.2):
        import mmapy

        self._mmasub = mmapy.mmasub
        self.n = int(n)
        self.move = move
        self.low = np.zeros((self.n, 1))
        self.upp = np.ones((self.n, 1))
        self.xold1 = None
        self.xold2 = None
        self.iter = 0

    def step(self, xval, f0val, df0dx, fval, dfdx):
        n = self.n
        xval = np.asarray(xval, dtype=float).reshape(n, 1)
        if self.xold1 is None:
            self.xold1 = self.xold2 = xval
        self.iter += 1
        xmma, *_, self.low, self.upp = self._mmasub(
            1,
            n,
            self.iter,
            xval,
            np.zeros((n, 1)),  # xmin
            np.ones((n, 1)),  # xmax
            self.xold1,
            self.xold2,
            f0val,
            np.asarray(df0dx, dtype=float).reshape(n, 1),
            np.asarray(fval, dtype=float).reshape(1, 1),
            np.asarray(dfdx, dtype=float).reshape(1, n),
            self.low,
            self.upp,
            1.0,  # a0
            np.zeros((1, 1)),  # a
            1e3 * np.ones((1, 1)),  # c
            np.zeros((1, 1)),  # d
            move=self.move,
        )
        self.xold2, self.xold1 = self.xold1, xval
        return xmma


class _StructuredFEM:
    """structured-grid assembly core shared by the topology optimization solvers.

    Holds the grid <-> element reshaping and the design-independent free-dof sparsity,
    so a SIMP loop assembles by gather + segmented sum into a fixed CSR. Subclasses add
    the factorization and solve. Physics enters through unit-material element matrices
    `local_mats` of shape (n_sub, ndof_e, ndof_e) and grid-shaped coefficient fields.

    Args:
        efts: element location maps, shape (n_elems, ndof_e).
        free: indices of the unconstrained dofs.
        ndof: total number of dofs.
        grid_shape: design-grid resolution per axis, 2D or 3D.
        sub_voxels: design sub-cells per element edge.
        upper: keep only the upper triangle (symmetric pardiso input), else the full
            symmetric matrix.
    """

    def __init__(self, efts, free, ndof, grid_shape, sub_voxels=1, upper=False):
        self.efts = efts
        self.grid_shape = tuple(int(s) for s in grid_shape)
        self.sub = int(sub_voxels)
        self.dim = len(self.grid_shape)
        self.nel = [s // self.sub for s in self.grid_shape]
        self.n_elems = int(np.prod(self.nel))
        self.n_sub = self.sub**self.dim
        self.nfree = free.size

        # every element drops ndof_e^2 entries onto the fixed free-dof sparsity; a 0/1
        # scatter matrix sums them into the CSR data, so assembly is gemm + one matvec
        ndof_e = efts.shape[1]
        dof_map = np.full(ndof, -1)
        dof_map[free] = np.arange(free.size)
        rows = np.repeat(dof_map[efts], ndof_e, axis=1).ravel()
        cols = np.tile(dof_map[efts], (1, ndof_e)).ravel()
        keep = (rows >= 0) & (cols >= 0) & ((rows <= cols) if upper else True)
        rows, cols = rows[keep], cols[keep]
        pattern = scipy.sparse.csr_matrix(
            (np.ones(rows.size), (rows, cols)), shape=(self.nfree, self.nfree)
        )
        pattern.sum_duplicates()  # canonical: sorted, unique column indices per row
        self._indices, self._indptr = pattern.indices, pattern.indptr
        data_rows = np.repeat(np.arange(self.nfree), np.diff(self._indptr))
        key = lambda r, c: r.astype(np.int64) * self.nfree + c  # row-major CSR order
        position = np.searchsorted(key(data_rows, self._indices), key(rows, cols))
        self._scatter = scipy.sparse.csr_matrix(
            (np.ones(position.size), (position, np.flatnonzero(keep))),
            shape=(self._indices.size, keep.size),
        )
        # data positions of the diagonal, for optional nodal springs (compliant
        # mechanisms): every free dof self-couples, so each diagonal entry exists
        self._diag_idx = np.flatnonzero(data_rows == self._indices)

    def grid_to_elements(self, field):  # grid -> (n_elems, n_sub)
        shape = [n for pair in zip(self.nel, [self.sub] * self.dim) for n in pair]
        perm = list(range(0, 2 * self.dim, 2)) + list(range(1, 2 * self.dim, 2))
        return field.reshape(shape).transpose(perm).reshape(self.n_elems, self.n_sub)

    def elements_to_grid(self, field):  # (n_elems, n_sub) -> grid
        shape = self.nel + [self.sub] * self.dim
        perm = [ax for d in range(self.dim) for ax in (d, self.dim + d)]
        return field.reshape(shape).transpose(perm).reshape(self.grid_shape)

    def data(self, coeff, local_mats):  # CSR values of per-voxel-scaled local_mats
        if np.iscomplexobj(coeff):  # two real passes beat upcasting the scatter matrix
            return self.data(coeff.real, local_mats) + 1j * self.data(
                coeff.imag, local_mats
            )
        scale_e = self.grid_to_elements(coeff)
        mat_e = scale_e @ np.asarray(local_mats).reshape(self.n_sub, -1)
        return self._scatter @ mat_e.ravel()

    def matrix(self, data):  # CSR on the fixed free-dof sparsity
        return scipy.sparse.csr_matrix(
            (data, self._indices, self._indptr), shape=(self.nfree, self.nfree)
        )

    def bilinear(self, local_mats, left, right):  # grid-shaped left^T local_mats right
        l_e, r_e = left[self.efts], right[self.efts]
        # one gemm beats np.einsum's 3-operand path ~3x: tmp[e,s,i] = K[s,i,j] r[e,j]
        flat = np.asarray(local_mats).transpose(2, 0, 1).reshape(r_e.shape[1], -1)
        tmp = (r_e @ flat).reshape(len(r_e), -1, l_e.shape[1])
        return self.elements_to_grid(np.einsum("esi,ei->es", tmp, l_e))

    def element_energy(self, u):  # grid-shaped unit-material energy u^T K_local u
        return self.bilinear(self.K_locals, u, u)


class StructuredFEM(_StructuredFEM):
    """real SPD assemble-and-solve on a structured grid, backed by MKL pardiso.

    Pardiso analyzes the sparsity once and only refactorizes (Cholesky on the upper
    triangle) as the design changes. `K_locals` are the unit-material element matrices,
    so the same class serves heat and elasticity. Pardiso runs on MKL threads, so
    drivers pin OPENBLAS to one thread. See `_StructuredFEM` for the shared arguments.
    """

    def __init__(self, efts, free, ndof, K_locals, grid_shape, sub_voxels=1):
        super().__init__(efts, free, ndof, grid_shape, sub_voxels, upper=True)
        from solvers.mklwrapper import pardisoFactorize

        self._pardiso = pardisoFactorize
        self._factor = None
        self.K_locals = K_locals

    def solve(self, material, rhs_free, spring_diag=None):  # K(material) u = rhs
        data = self.data(material, self.K_locals)
        if spring_diag is not None:  # add nodal springs onto the diagonal (mechanisms)
            data[self._diag_idx] += spring_diag
        if self._factor is None:
            self._factor = self._pardiso(
                self.matrix(data), symmetric=True, symmetricHalf=True
            )
        else:
            self._factor.refactorize(self.matrix(data))
        return self._factor(np.asarray(rhs_free, dtype=float))


class CGStructuredFEM(_StructuredFEM):
    """drop-in `StructuredFEM` variant solved with Jacobi-preconditioned CG.

    Factorization-free and warm-started from the previous design's solution; the
    iteration count grows with the SIMP contrast, so the direct solvers stay faster on
    small 2D problems. See `StructuredFEM` for the shared arguments.

    Args:
        use_cupy: solve on the GPU with cupyx CG instead of mlhp CG on the CPU.
        rtol: relative residual tolerance of the CG solve.
        maxiter: CG iteration cap per solve.
    """

    def __init__(
        self,
        efts,
        free,
        ndof,
        K_locals,
        grid_shape,
        sub_voxels=1,
        use_cupy=False,
        rtol=1e-8,
        maxiter=10000,
    ):
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
            self._cpx_linalg = cupyx.scipy.sparse.linalg
            self._K_gpu = cupyx.scipy.sparse.csr_matrix(
                (
                    cupy.zeros(self._indices.size),
                    cupy.asarray(self._indices),
                    cupy.asarray(self._indptr),
                ),
                shape=(self.nfree, self.nfree),
            )
        else:
            import mlhp

            self._mlhp = mlhp

    def solve(self, material, rhs_free, spring_diag=None):  # K(material) u = rhs
        data = self.data(material, self.K_locals)
        if spring_diag is not None:  # add nodal springs onto the diagonal (mechanisms)
            data[self._diag_idx] += spring_diag
        diag = data[self._diag_idx]
        rhs = np.asarray(rhs_free, dtype=float)
        if self.use_cupy:
            cp = self._cp
            self._K_gpu.data[:] = cp.asarray(data)
            inv_diag = 1.0 / cp.asarray(diag)
            M = self._cpx_linalg.LinearOperator(
                self._K_gpu.shape, matvec=lambda x: inv_diag * x
            )
            u, info = self._cpx_linalg.cg(
                self._K_gpu,
                cp.asarray(rhs),
                x0=cp.asarray(self._x0),
                rtol=self.rtol,
                maxiter=self.maxiter,
                M=M,
            )
            u = cp.asnumpy(u)
        else:
            mlhp = self._mlhp
            K = self.matrix(data)
            A = mlhp.linearOperator_array(lambda x, out: np.copyto(out, K @ x))
            M = mlhp.linearOperator_array(lambda x, out: np.divide(x, diag, out=out))
            u = mlhp.cg(
                A,
                mlhp.DoubleVector(rhs.tolist()),
                x0=mlhp.DoubleVector(self._x0.tolist()),
                rtol=self.rtol,
                maxiter=self.maxiter,
                M=M,
            )
            u = np.array(u.array)
        self._x0 = u
        return u


class ComplexStructuredFEM(_StructuredFEM):
    """complex-symmetric Helmholtz assemble-and-solve on a structured grid (pardiso).

    The system `S = a K + b M` mixes stiffness and mass element matrices, each scaled
    by its own per-voxel coefficient field, with `b` allowed complex (damping). Only
    the upper triangle is stored, as pardiso's complex-symmetric input. The caller
    builds the constant `dS/dvar` from `K_locals` / `M_locals` and feeds it to
    `bilinear` for the adjoint sensitivity. See `_StructuredFEM` for the shared
    arguments.
    """

    def __init__(self, efts, free, ndof, K_locals, M_locals, grid_shape, sub_voxels=1):
        super().__init__(efts, free, ndof, grid_shape, sub_voxels, upper=True)
        from solvers.mklwrapper import pardisoFactorize

        self._pardiso = pardisoFactorize
        self._factor = None
        self.K_locals = K_locals
        self.M_locals = M_locals

    def system(self, coeff_k, coeff_m):  # upper triangle of a(x) K + b(x) M, CSR
        return self.matrix(
            self.data(coeff_k, self.K_locals) + self.data(coeff_m, self.M_locals)
        )

    def factorize(self, matrix):  # pardiso refactorization on the fixed sparsity
        if self._factor is None:
            self._factor = self._pardiso(matrix, symmetric=True, symmetricHalf=True)
        else:
            self._factor.refactorize(matrix)
        return self._factor


class DensityFilter:
    """conic density filter of radius `rmin` in 2D or 3D, with adjoint and OC variants.

    Pass `xp=cupy` (and optionally a `dtype`) to build and convolve on the GPU, and
    `fft=True` for large radii, where the direct convolution costs O(rmin^2) per cell.

    References:
        https://doi.org/10.1007/s001580050176
        https://doi.org/10.1002/nme.116
    """

    def __init__(self, rmin, shape, xp=np, dtype=None, fft=False):
        if xp is np:
            import scipy.ndimage as ndi
            import scipy.signal as signal
        else:
            import cupyx.scipy.ndimage as ndi
            import cupyx.scipy.signal as signal
        self.xp, self.ndi, self.signal, self.fft = xp, ndi, signal, fft
        r = xp.arange(-math.ceil(rmin), math.ceil(rmin) + 1)
        grid = xp.meshgrid(*[r] * len(shape), indexing="ij")
        self.kernel = xp.maximum(0.0, rmin - xp.sqrt(sum(g**2 for g in grid)))
        if dtype is not None:
            self.kernel = self.kernel.astype(dtype)
        self.Hs = self._convolve(xp.ones(shape, self.kernel.dtype))

    def _convolve(self, x):
        if self.fft:
            return self.signal.fftconvolve(x, self.kernel, mode="same")
        return self.ndi.convolve(x, self.kernel, mode="constant", cval=0.0)

    def __call__(self, x):  # conic smoothing of the raw design
        return self._convolve(x) / self.Hs

    def adjoint(self, g):  # transpose of the filter, for the chain rule
        return self._convolve(g / self.Hs)

    def sensitivity(self, rho, dc):  # classic OC sensitivity filter
        return self._convolve(rho * dc) / (self.xp.maximum(rho, 1e-3) * self.Hs)


def simp(x, penal, v_min, v_max):
    """solid isotropic material with penalization, interpolating v_min to v_max.

    Reference: https://doi.org/10.1007/BF01650949
    """
    return v_min + x**penal * (v_max - v_min)


def dsimp(x, penal, v_min, v_max):
    """derivative of `simp` with respect to x."""
    return penal * x ** (penal - 1) * (v_max - v_min)


def projection(x, beta, eta):
    """smoothed Heaviside projection about the threshold eta (NumPy or CuPy x).

    Reference: https://doi.org/10.1007/s00158-010-0602-y
    """
    a, b = math.tanh(beta * eta), math.tanh(beta * (1.0 - eta))
    return (a + np.tanh(beta * (x - eta))) / (a + b)


def dprojection(x, beta, eta):
    """derivative of `projection` with respect to x."""
    a, b = math.tanh(beta * eta), math.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (x - eta)) ** 2) / (a + b)
