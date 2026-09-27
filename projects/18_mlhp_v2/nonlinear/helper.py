"""matrix-free p/h-multigrid preconditioned CG for voxel finite elements, with operator
families and stored element tangents for nonlinear problems."""

import copy

import cupy as cp
import cupyx.scipy.sparse
import mlhp
import numpy as np
import scipy.sparse

# one thread per row sums over the adjacent elements: no atomics, reproducible sums
KERNELS = r"""
template <typename T, int N, int S, int M>
__device__ __forceinline__ T row(const long long i, const T* coeff, const T* K,
                                 const int* efts, const int* adj, const T* mask,
                                 const T* u)
{
    if (mask[i] == 0) return u[i];
    T acc = 0;
    #pragma unroll
    for (int k = 0; k < M; ++k) {
        const int code = adj[i * M + k];
        if (code < 0) break;
        const long long e = code / N;
        const int l = code % N;
        const int* eft = efts + e * N;
        if (S == 0) {
            const T* Kl = K + (e * N + l) * N;
            #pragma unroll
            for (int j = 0; j < N; ++j) acc += Kl[j] * u[eft[j]];
        } else {
            T ue[N];
            #pragma unroll
            for (int j = 0; j < N; ++j) ue[j] = u[eft[j]];
            for (int s = 0; s < S; ++s) {
                const T* Kl = K + (s * N + l) * N;
                T sum = 0;
                #pragma unroll
                for (int j = 0; j < N; ++j) sum += Kl[j] * ue[j];
                acc += coeff[e * S + s] * sum;
            }
        }
    }
    return acc;
}

#define ROW_KERNEL(name, args, body)                                                 \
template <typename T, int N, int S, int M>                                          \
__global__ void name(const T* coeff, const T* K, const int* efts, const int* adj,   \
                     const T* mask, const int ndof, args)                           \
{                                                                                    \
    const long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;           \
    if (i >= ndof) return;                                                           \
    body                                                                             \
}
#define ROW(u) row<T, N, S, M>(i, coeff, K, efts, adj, mask, u)
#define COMMA ,

// out = A u
ROW_KERNEL(apply, const T* u COMMA T* out, out[i] = ROW(u);)

// r = b - A x, d = dinv r / theta
ROW_KERNEL(residual,
           const T* b COMMA const T* x COMMA const T* dinv COMMA const T* cheb
           COMMA T* r COMMA T* d,
           r[i] = b[i] - ROW(x); d[i] = dinv[i] * r[i] * cheb[0];)

// r -= A d
ROW_KERNEL(update_residual, const T* d COMMA T* r, r[i] -= ROW(d);)

// Chebyshev step: r -= A d_in, d_out = c1 d_in + c2 dinv r, x += d_out (+ d_in)
ROW_KERNEL(chebyshev,
           const T* dinv COMMA const T* cheb COMMA const T* d_in COMMA T* d_out
           COMMA T* r COMMA T* x COMMA const int add_in,
           const T ri = r[i] - ROW(d_in); r[i] = ri;
           const T dn = cheb[0] * d_in[i] + cheb[1] * dinv[i] * ri;
           d_out[i] = dn; x[i] += add_in ? dn + d_in[i] : dn;)

template <typename T>
__global__ void csr_matvec(const int* indptr, const int* indices, const T* data,
                           const T* x, T* y, const T* mask, const int rows)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= rows) return;
    T acc = 0;
    for (int k = indptr[i]; k < indptr[i + 1]; ++k) acc += data[k] * x[indices[k]];
    y[i] = mask ? y[i] + mask[i] * acc : acc;
}

template <typename T>
__global__ void dense_matvec(const double* A, const T* x, T* y, const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double acc = 0;
    for (int j = 0; j < n; ++j) acc += A[(long long)i * n + j] * (double)x[j];
    y[i] = (T)acc;
}

__global__ void dot_partial(const double* a, const double* b, double* partial,
                            const long long n)
{
    __shared__ double cache[256];
    double acc = 0;
    for (long long k = blockIdx.x * 256 + threadIdx.x; k < n; k += 256LL * gridDim.x)
        acc += a[k] * b[k];
    cache[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) cache[threadIdx.x] += cache[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = cache[0];
}

__global__ void dot_final(const double* partial, double* out, const int blocks)
{
    double acc = 0;
    for (int k = 0; k < blocks; ++k) acc += partial[k];
    *out = acc;
}
"""
DOT_BLOCKS = 128
_functions = {}


def kernel(name):
    """compiled template instance, e.g. `kernel("apply<float, 8, 1, 4>")`."""
    if name not in _functions:
        module = cp.RawModule(code=KERNELS, name_expressions=[name])
        _functions[name] = module.get_function(name)
    return _functions[name]


def ctype(dtype):
    return "float" if dtype == cp.float32 else "double"


def launch(function, threads, args):
    function(((threads + 255) // 256,), (256,), args)


_pre_first = cp.ElementwiseKernel(
    "T b, T dinv, raw T cheb",
    "T r, T d, T x",
    "r = b; d = dinv * b * cheb[0]; x = d",
    "pre_first",
)
_gather_rows = cp.ElementwiseKernel(
    "raw int32 adj, raw T values, int32 M",
    "T out",
    "T acc = 0; for (int k = 0; k < M; ++k) { const int c = adj[i * M + k]; "
    "if (c >= 0) acc += values[c]; } out = acc",
    "gather_rows",
)
_add = cp.ElementwiseKernel("T a", "T x", "x += a", "add")
_cast = cp.ElementwiseKernel("S a", "T b", "b = (T)a", "cast")
_xr_update = cp.ElementwiseKernel(
    "T p, T Ap, raw T s",
    "T x, T r",
    "T alpha = s[0] / s[1]; x += alpha * p; r -= alpha * Ap",
    "xr_update",
)
_p_update = cp.ElementwiseKernel(
    "T z, raw T s", "T p", "p = z + (s[4] - s[3]) / s[0] * p", "p_update"
)


def grid_to_elements(field, nel, sub):
    """voxel grid -> (n_elems, sub^D) in C order of elements and sub-voxels."""
    D = len(nel)
    shape = [n for pair in zip(nel, [sub] * D) for n in pair]
    perm = list(range(0, 2 * D, 2)) + list(range(1, 2 * D, 2))
    return field.reshape(shape).transpose(perm).reshape(int(np.prod(nel)), sub**D)


def elements_to_grid(field, nel, sub):
    """(n_elems, sub^D) -> voxel grid, the inverse of `grid_to_elements`."""
    D = len(nel)
    perm = [ax for d in range(D) for ax in (d, D + d)]
    grid = field.reshape(list(nel) + [sub] * D).transpose(perm)
    return grid.reshape([n * sub for n in nel])


def basis_values(basis, points, difforder=0, icomponent=0):
    """(n_points, ndof) values (or derivatives along icomponent) of every basis function."""
    values = []
    for j in range(basis.ndof()):
        dofs = np.zeros(basis.ndof())
        dofs[j] = 1.0
        evaluator = mlhp.scalarEvaluator(
            basis, mlhp.DoubleVector(dofs), difforder=difforder, icomponent=icomponent
        )
        values.append(np.array(evaluator(*points)))
    return np.array(values).T


def isotropic_tensor(D, lam, mu):
    """(D^2, D^2) small strain elasticity acting on the displacement gradient."""
    eye = np.eye(D)
    C = lam * np.einsum("ij,kl->ijkl", eye, eye)
    C += mu * (np.einsum("ik,jl->ijkl", eye, eye) + np.einsum("il,jk->ijkl", eye, eye))
    return C.reshape(D * D, D * D)


class QuadratureTables:
    """shape functions at the Gauss points of every sub-voxel of one fine element.

    All elements of the voxel grid are translates of each other, so these tables hold
    the whole geometry: any quadratic form in the solution and its gradient with a
    per sub-voxel constant (or per point) material becomes a small contraction.
    Element dofs are ordered by field, then by scalar basis function.
    """

    def __init__(self, space, cell, degree, sub_voxels, order=None):
        D = len(cell)
        mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=cell))
        basis = space(mesh, degree=degree, nfields=1)
        assert np.array_equal(
            np.array(basis.locationMaps())[0], np.arange(basis.ndof())
        )
        gauss, weights = np.polynomial.legendre.leggauss(
            degree + 1 if order is None else order
        )
        size = np.asarray(cell) / sub_voxels
        corners = np.indices([sub_voxels] * D).reshape(D, -1).T * size  # (S, D)
        local = np.stack(np.meshgrid(*[gauss] * D, indexing="ij"), -1).reshape(-1, D)
        points = corners[:, None, :] + 0.5 * (local + 1.0) * size  # (S, Q, D)
        flat = [points[..., d].ravel() for d in range(D)]
        S, Q = points.shape[:2]
        self.N = basis_values(basis, flat).reshape(S, Q, -1)
        self.dN = np.stack(
            [basis_values(basis, flat, 1, d) for d in range(D)], -1
        ).reshape(S, Q, -1, D)
        w = np.prod(np.stack(np.meshgrid(*[weights] * D, indexing="ij"), -1), -1)
        self.w = np.tile(w.ravel() * np.prod(0.5 * size), (S, 1))
        self.volume = self.w.sum(axis=1)
        self.D, self.n = D, self.N.shape[2]

    def gradient(self, nfields):
        """(S, Q, nfields * D, nfields * n): element dofs -> du_i / dx_j at i * D + j."""
        return np.einsum("fg,sqaj->sqfjga", np.eye(nfields), self.dN).reshape(
            *self.w.shape, nfields * self.D, nfields * self.n
        )

    def value(self, nfields):
        """(S, Q, nfields, nfields * n): element dofs -> u_i."""
        return np.einsum("fg,sqa->sqfga", np.eye(nfields), self.N).reshape(
            *self.w.shape, nfields, nfields * self.n
        )

    def integrate(self, op, C=None):
        """(S, n, n) sub-voxel matrices of the quadratic form op^T C op."""
        Cop = op if C is None else np.einsum("ij,sqjk->sqik", C, op)
        return np.einsum("sq,sqia,sqib->sab", self.w, op, Cop)

    def mean(self, op):
        """(S, ...) sub-voxel average of a pointwise operator (S, Q, ...)."""
        return np.einsum("sq,sq...->s...", self.w, op) / self.volume.reshape(
            -1, *[1] * (op.ndim - 2)
        )

    def at_points(self, op, u_e):
        """(n_elems, S * Q, rows) pointwise op u_e of element vectors u_e (n_elems, n)."""
        flat = cp.asarray(op.reshape(-1, op.shape[-1]))
        return (u_e @ flat.T).reshape(len(u_e), -1, op.shape[-2])

    def forces(self, op, flux):
        """(n_elems, n) sum_p w_p op_p^T flux_p of a pointwise flux (n_elems, S * Q, rows)."""
        weighted = cp.asarray((self.w[..., None, None] * op).reshape(-1, op.shape[-1]))
        return flux.reshape(len(flux), -1) @ weighted

    def tangents(self, op, A):
        """(n_elems, n, n) sum_p w_p op_p^T A_p op_p of pointwise A (n_elems, S * Q, rows, rows).

        Computed in the precision of A.
        """
        ops = cp.asarray(op.reshape(-1, *op.shape[-2:]), dtype=A.dtype)
        weighted = cp.asarray(
            (self.w.reshape(-1, 1, 1) * op.reshape(-1, *op.shape[-2:])), dtype=A.dtype
        ).reshape(-1, op.shape[-1])
        return weighted.T @ (A @ ops).reshape(len(A), -1, op.shape[-1])


MATERIAL = r"""
template <int D, int NP>
__global__ void evaluate_material(const double* H, const double* prm, double* P,
                                  double* A, const int Q, const long long npoints)
{
    const long long p = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= npoints) return;
    material<D>(H + p * D * D, prm + (p / Q) * NP, P + p * D * D, A + p * D * D * D * D);
}
"""


class PointMaterial:
    """GPU material subroutine: first Piola stress and tangent from the displacement gradient.

    `code` defines `template <int D> __device__ void material(const double* H,
    const double* prm, double* P, double* A)` with H = grad u, P (D^2) and
    A = dP/dF (D^2 x D^2), all row-major. prm points to the parameters of the
    sub-voxel holding the point.
    """

    def __init__(self, code, D, nparams):
        name = f"evaluate_material<{D}, {nparams}>"
        module = cp.RawModule(code=code + MATERIAL, name_expressions=[name])
        self.function, self.D = module.get_function(name), D

    def __call__(self, H, params):
        """H (n_elems, S * Q, D^2) and params (n_elems, S, nparams) -> (P, A)."""
        D2 = self.D**2
        Q = H.shape[1] // params.shape[1]
        P = cp.empty(H.shape)
        A = cp.empty((*H.shape[:2], D2, D2))
        n = H.shape[0] * H.shape[1]
        args = (cp.ascontiguousarray(H), cp.ascontiguousarray(params), P, A)
        launch(self.function, n, args + (np.int32(Q), np.int64(n)))
        return P, A


def local_transfers(space, D, degree_fine, degree_coarse, children):
    """(children^D, nf, nc) maps from a coarse element's dofs to each child element's.

    Fits the coarse basis in the fine one by least squares on sample points, which is
    exact for nested spaces and independent of the shape function family.
    """
    unit = [1.0] * D
    fine_mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[children] * D, lengths=unit))
    coarse_mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=unit))
    fine = space(fine_mesh, degree=degree_fine, nfields=1)
    coarse = space(coarse_mesh, degree=degree_coarse, nfields=1)
    samples = 3 * (degree_fine + 1) * children
    axis = (np.arange(samples) + 0.5) / samples
    points = [g.ravel() for g in np.meshgrid(*[axis] * D, indexing="ij")]
    T = np.linalg.lstsq(
        basis_values(fine, points), basis_values(coarse, points), rcond=None
    )[0]
    T[np.abs(T) < 1e-12] = 0.0
    return T[np.array(fine.locationMaps())]


def assign(obj, name, value):
    """store value in place when possible, so captured CUDA graphs stay valid."""
    old = getattr(obj, name, None)
    same = old is not None and value is not None and old.shape == value.shape
    if same and old.dtype == value.dtype:
        old[...] = value
    else:
        setattr(obj, name, value)


class Level:
    """one grid of the hierarchy, applied row by row without a global matrix.

    Element matrices are either sum_s coeff[e, s] K[s] (voxel coefficients times
    preintegrated sub-voxel matrices) or stored per element (Galerkin coarse levels).
    Fixed dofs act as identity rows, A = M K M + (I - M); vectors passed to the row
    kernels must vanish on fixed dofs.
    """

    def __init__(self, basis, nel, degree, dtype):
        self.basis, self.nel, self.degree, self.dtype = basis, list(nel), degree, dtype
        self.ndof = basis.ndof()
        efts = np.array(basis.locationMaps())
        self.efts = cp.asarray(efts, dtype=cp.int32)
        self.n_elems, self.n = efts.shape

        # dof -> (element * n + local index) of every element holding it, -1 padded
        dofs = efts.ravel()
        order = np.argsort(dofs, kind="stable")
        counts = np.bincount(dofs, minlength=self.ndof)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        adj = np.full((self.ndof, counts.max()), -1, dtype=np.int32)
        adj[dofs[order], np.arange(dofs.size) - starts[dofs[order]]] = order
        self.adj = cp.asarray(adj)

        self.b, self.x, self.r, self.d, self.d2 = (
            cp.zeros(self.ndof, dtype) for _ in range(5)
        )
        self._power_vector = None

    def set_operator(self, K, coeff=None, mask=None):
        """stored element matrices K (n_elems, n, n), or coeff (n_elems, S) with K (S, n, n)."""
        assign(self, "K", K.astype(self.dtype))
        assign(self, "coeff", None if coeff is None else coeff.astype(self.dtype))
        diag_e = cp.diagonal(K, axis1=1, axis2=2)
        diag_e = (diag_e if coeff is None else coeff @ diag_e).ravel()
        diag = cp.empty(self.ndof, diag_e.dtype)
        _gather_rows(self.adj, diag_e, self.adj.shape[1], diag)
        mask = (diag > 0) if mask is None else mask
        assign(self, "mask", mask.astype(self.dtype))
        dinv = 1.0 / (diag * self.mask + (1.0 - self.mask))
        assign(self, "dinv", dinv.astype(self.dtype))
        self.S = 0 if coeff is None else coeff.shape[1]

    def row_kernel(self, name, *args, mask=None):
        template = (
            f"{name}<{ctype(self.dtype)}, {self.n}, {self.S}, {self.adj.shape[1]}>"
        )
        common = (
            self.K if self.S == 0 else self.coeff,
            self.K,
            self.efts,
            self.adj,
            self.mask if mask is None else mask,
            np.int32(self.ndof),
        )
        launch(kernel(template), self.ndof, common + args)

    def apply(self, u, out, full=False):
        """out = A u, or the unconstrained K u (for lifting and reactions) if full."""
        if full and getattr(self, "_ones", None) is None:
            self._ones = cp.ones(self.ndof, self.dtype)
        self.row_kernel("apply", u, out, mask=self._ones if full else None)
        return out

    def estimate_lmax(self, iters):  # power iteration on D^-1 A, warm-started
        if self._power_vector is None:
            rng = cp.random.default_rng(0)
            self._power_vector = rng.random(self.ndof).astype(self.dtype)
            iters *= 3
        v = self._power_vector * self.mask
        Av = cp.empty_like(v)
        for _ in range(iters):
            v = self.dinv * self.apply(v / cp.linalg.norm(v), Av)
        self._power_vector = v
        return float(cp.linalg.norm(v))

    def dense(self):  # float64 dense matrix of a stored-matrix level
        K_e = cp.asnumpy(self.K).astype(np.float64)
        efts = cp.asnumpy(self.efts)
        rows = np.repeat(efts, self.n, axis=1).ravel()
        cols = np.tile(efts, (1, self.n)).ravel()
        dense = scipy.sparse.coo_matrix(
            (K_e.ravel(), (rows, cols)), shape=(self.ndof, self.ndof)
        ).toarray()
        dense[np.diag_indices(self.ndof)] += 1.0 - cp.asnumpy(self.mask)
        return dense


class VoxelMultigrid:
    """multigrid-preconditioned CG for voxel elements with a coefficient-linear operator.

    The fine element matrix is either sum_t sum_s c[t, e, s] K_t[s], a few operator
    families preintegrated per sub-voxel and scaled by voxel coefficients
    (`update`), or stored per element (`update_elements`, e.g. tangents of a
    general material). The hierarchy drops to p = 1 on the same mesh, then halves the
    mesh until a dense solve is cheap. Coarse element matrices are Galerkin products
    of their children, so the V-cycle follows high material contrast. One CG
    iteration, V-cycle included, is replayed as a single CUDA graph.

    Args:
        families: list of (S, n, n) unit-coefficient sub-voxel matrices (see
            `QuadratureTables`), or None when only `update_elements` is used.
        nvoxels, lengths: voxel grid resolution and domain size.
        degree, sub_voxels, nfields: fine discretization.
        dirichlet: callable mapping the fine mlhp basis to its fixed dof indices.
        space: mlhp basis factory (`makeHpTensorSpace` or `makeHpTrunkSpace`).
        smoothing: Chebyshev degree (matvecs) of each pre- and post-smoothing, one int
            for all levels or a list from fine to coarse (last entry repeats).
        coarse_dofs: stop coarsening once a level has at most this many dofs.
        dtype: precision of the V-cycle; the outer CG always runs in float64.
    """

    def __init__(
        self,
        families,
        nvoxels,
        lengths,
        degree,
        sub_voxels,
        nfields,
        dirichlet,
        space=mlhp.makeHpTensorSpace,
        smoothing=(2, 4, 8, 16),
        coarse_dofs=300,
        dtype=cp.float32,
        eig_ratio=30.0,
        power_iters=8,
    ):
        D = len(nvoxels)
        self.D, self.sub, self.dtype = D, sub_voxels, dtype
        self.eig_ratio, self.power_iters = eig_ratio, power_iters
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.graph = None
        nel = [n // sub_voxels for n in nvoxels]
        assert all(n * sub_voxels == v for n, v in zip(nel, nvoxels))
        self.nvoxels = list(nvoxels)

        def make_level(nel, degree):
            mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nel, lengths=lengths))
            return Level(
                space(mesh, degree=degree, nfields=nfields), nel, degree, dtype
            )

        self.K_locals = None
        if families is not None:
            self.families = len(families)
            self.K_locals = cp.asarray(np.concatenate(families))

        self.levels = [make_level(nel, degree)]
        self.operator = copy.copy(self.levels[0])  # float64 fine level for the outer CG
        self.operator.dtype = cp.float64
        self.basis, self.ndof = self.operator.basis, self.operator.ndof
        self.fixed = np.asarray(dirichlet(self.basis), dtype=np.int64)
        self.mask = cp.ones(self.ndof)
        self.mask[cp.asarray(self.fixed)] = 0.0

        self.transfers = []
        while True:
            fine = self.levels[-1]
            if fine.degree > 1:
                coarse, children = make_level(fine.nel, 1), 1
            elif all(n % 2 == 0 for n in fine.nel) and fine.ndof > coarse_dofs:
                coarse, children = make_level([n // 2 for n in fine.nel], 1), 2
            else:
                break
            T = local_transfers(space, D, fine.degree, 1, children)
            T = np.stack([np.kron(np.eye(nfields), Tk) for Tk in T])
            self.transfers.append(self._transfer(fine, coarse, T, children))
            self.levels.append(coarse)
        smoothing = [smoothing] if np.isscalar(smoothing) else list(smoothing)
        for i, lv in enumerate(self.levels):
            lv.smoothing = smoothing[min(i, len(smoothing) - 1)]
            lv.cheb = cp.zeros(2 * lv.smoothing - 1, dtype)

        if self.transfers:
            tr = self.transfers[0]
            fixed_e = cp.any(self.mask[self.levels[0].efts] == 0.0, axis=1)
            self._boundary = cp.flatnonzero(cp.any(fixed_e[tr["children"]], axis=1))
            # level 1 straight from the voxel coefficients, see the README
            if self.K_locals is not None:
                K_s = self.K_locals.reshape(len(self.K_locals), -1)
                self._K1_locals = cp.concatenate(
                    [K_s @ G for G in tr["G_blocks"]]
                ).astype(dtype)

        # persistent CG state; scalars hold rz, pAp, rr, rz_old, rz_new
        self.cg = {name: cp.zeros(self.ndof) for name in ("x", "r", "z", "p", "Ap")}
        self.scalars = cp.zeros(5)
        self.partial = cp.zeros(DOT_BLOCKS)

    def _transfer(self, fine, coarse, T, children):
        # parent element and child position of every fine element (C order)
        ids = np.indices(fine.nel).reshape(self.D, -1)
        parent = np.ravel_multi_index(tuple(ids // children), coarse.nel)
        position = np.ravel_multi_index(tuple(ids % children), [children] * self.D)
        efts_f, efts_c = fine.efts.get(), coarse.efts.get()
        vals = T[position]
        rows = np.broadcast_to(efts_f[:, :, None], vals.shape).ravel()
        cols = np.broadcast_to(efts_c[parent][:, None, :], vals.shape).ravel()
        vals = vals.ravel()
        keep = vals != 0.0
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        # dofs shared by neighbouring children appear once per child, keep one copy
        _, first = np.unique(
            rows.astype(np.int64) * coarse.ndof + cols, return_index=True
        )
        P = scipy.sparse.csr_matrix(
            (vals[first], (rows[first], cols[first])), shape=(fine.ndof, coarse.ndof)
        )
        children_of = np.empty((coarse.n_elems, children**self.D), dtype=np.int64)
        children_of[parent, position] = np.arange(fine.n_elems)
        # Galerkin sum_k T_k^T K_k T_k as one gemm: vec(T^T K T) = kron(T, T)^T vec(K)
        G_blocks = [cp.asarray(np.kron(Tk, Tk)) for Tk in T]
        return {
            "P": cupyx.scipy.sparse.csr_matrix(P.astype(self.dtype)),
            "R": cupyx.scipy.sparse.csr_matrix(P.T.tocsr().astype(self.dtype)),
            "G": cp.concatenate(G_blocks).astype(self.dtype),
            "G_blocks": G_blocks,
            "children": cp.asarray(children_of),
        }

    def to_elements(self, grids):
        """(T, *nvoxels) voxel grids -> (n_elems, T * S) family-major element layout."""
        grids = cp.asarray(grids, dtype=cp.float64).reshape(-1, *self.nvoxels)
        nel = self.levels[0].nel
        return cp.concatenate([grid_to_elements(g, nel, self.sub) for g in grids], 1)

    def update(self, coeff, hierarchy=True):
        """set the operator for new voxel coefficients (T, *nvoxels).

        With hierarchy=False only the float64 CG operator changes and the V-cycle of
        the last full update is kept, which flexible CG tolerates while the operator
        changes little (a staggered or Newton loop).
        """
        cp.cuda.get_current_stream().synchronize()
        with self.stream:
            self._update(self.to_elements(coeff), None, hierarchy)
        self.stream.synchronize()

    def update_elements(self, K, hierarchy=True):
        """as `update`, for stored fine element matrices (n_elems, n, n)."""
        cp.cuda.get_current_stream().synchronize()
        with self.stream:
            self._update(None, cp.asarray(K, dtype=cp.float64), hierarchy)
        self.stream.synchronize()

    def _update(self, coeff, K_elements, hierarchy):
        if getattr(self, "_stored", None) != (coeff is None):
            self.graph = None  # kernels differ per mode
            self._stored = coeff is None
            hierarchy = True
        if coeff is not None:
            self.operator.set_operator(self.K_locals, coeff, self.mask)
        else:
            self.operator.set_operator(K_elements, None, self.mask)
        self.stale = not hierarchy
        if hierarchy:
            self._setup()

    def setup(self):
        """rebuild the V-cycle hierarchy from the current float64 operator."""
        self.stale = False
        self._setup()

    def _setup(self):  # V-cycle hierarchy from the current CG operator
        fine, op = self.levels[0], self.operator
        fine.set_operator(op.K, op.coeff, self.mask)
        for i, lv in enumerate(self.levels):
            hi = 1.1 * lv.estimate_lmax(self.power_iters)
            lo = hi / self.eig_ratio
            theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
            sigma = theta / delta
            rho = 1.0 / sigma
            cheb = [1.0 / theta]
            for _ in range(lv.smoothing - 1):
                rho_new = 1.0 / (2.0 * sigma - rho)
                cheb += [rho_new * rho, 2.0 * rho_new / delta]
                rho = rho_new
            lv.cheb[...] = cp.asarray(cheb, dtype=self.dtype)
            if i == len(self.transfers):
                break
            tr = self.transfers[i]
            if i == 0 and fine.S > 0:
                K = (
                    fine.coeff[tr["children"]].reshape(len(tr["children"]), -1)
                    @ self._K1_locals
                )
                ids = self._boundary
                K_children = cp.einsum(
                    "bks,sij->bkij", fine.coeff[tr["children"][ids]], fine.K
                )
            else:
                K = lv.K[tr["children"]].reshape(len(tr["children"]), -1) @ tr["G"]
                ids = self._boundary if i == 0 else None
                K_children = None if i else fine.K[tr["children"][ids]]
            if i == 0 and len(
                ids
            ):  # children coupled to fixed dofs stay out of level 1
                mask_e = fine.mask[fine.efts[tr["children"][ids]]]
                K_children *= mask_e[..., :, None] * mask_e[..., None, :]
                K[ids] = K_children.reshape(len(ids), -1) @ tr["G"]
            n = self.levels[i + 1].n
            self.levels[i + 1].set_operator(K.reshape(-1, n, n))
        assign(
            self, "coarse_inverse", cp.linalg.inv(cp.asarray(self.levels[-1].dense()))
        )

    def _smooth(self, lv, pre):
        """Chebyshev-Jacobi on lv.x for rhs lv.b; pre-smoothing starts from x = 0."""
        d, d_next = lv.d, lv.d2
        if pre:
            _pre_first(lv.b, lv.dinv, lv.cheb, lv.r, d, lv.x)
        else:
            lv.row_kernel("residual", lv.b, lv.x, lv.dinv, lv.cheb, lv.r, d)
            if lv.smoothing == 1:
                _add(d, lv.x)
        for k in range(lv.smoothing - 1):
            add_in = np.int32(not pre and k == 0)
            lv.row_kernel(
                "chebyshev",
                lv.dinv,
                lv.cheb[2 * k + 1 :],
                d,
                d_next,
                lv.r,
                lv.x,
                add_in,
            )
            d, d_next = d_next, d
        if pre:
            lv.row_kernel("update_residual", d, lv.r)

    def _csr(self, A, x, y, mask=None):  # y = A x, or y += mask * (A x)
        launch(
            kernel(f"csr_matvec<{ctype(self.dtype)}>"),
            A.shape[0],
            (
                A.indptr,
                A.indices,
                A.data,
                x,
                y,
                np.uint64(0) if mask is None else mask,
                np.int32(A.shape[0]),
            ),
        )

    def _vcycle(self):  # levels[0].b -> levels[0].x
        L = len(self.transfers)
        for i in range(L):
            self._smooth(self.levels[i], pre=True)
            self._csr(self.transfers[i]["R"], self.levels[i].r, self.levels[i + 1].b)
        coarse = self.levels[L]
        launch(
            kernel(f"dense_matvec<{ctype(self.dtype)}>"),
            coarse.ndof,
            (self.coarse_inverse, coarse.b, coarse.x, np.int32(coarse.ndof)),
        )
        for i in reversed(range(L)):
            lv = self.levels[i]
            self._csr(self.transfers[i]["P"], self.levels[i + 1].x, lv.x, lv.mask)
            self._smooth(lv, pre=False)

    def _dot(self, a, b, out):  # deterministic two-stage reduction
        partial = kernel("dot_partial")
        partial((DOT_BLOCKS,), (256,), (a, b, self.partial, np.int64(a.size)))
        final = kernel("dot_final")
        final((1,), (1,), (self.partial, out, np.int32(DOT_BLOCKS)))

    def _precondition(self, r, z):
        _cast(r, self.levels[0].b)
        self._vcycle()
        _cast(self.levels[0].x, z)

    def _iteration(self):  # one flexible PCG step on the persistent state
        x, r, z, p, Ap = (self.cg[k] for k in ("x", "r", "z", "p", "Ap"))
        s = self.scalars
        self.operator.apply(p, Ap)
        self._dot(p, Ap, s[1])
        _xr_update(p, Ap, s, x, r)
        self._dot(r, r, s[2])
        self._dot(r, z, s[3])
        self._precondition(r, z)
        self._dot(r, z, s[4])
        _p_update(z, s, p)  # flexible beta, the float32 V-cycle is inexact
        s[0:1] = s[4:5]

    def solve(self, rhs, x0=None, rtol=1e-8, maxiter=500, atol=0.0, refresh=20):
        """flexible PCG on the float64 fine operator; returns (u, iterations).

        Fixed dofs of rhs and x0 are ignored, the solution vanishes there. A V-cycle
        kept stale by `update(..., hierarchy=False)` is rebuilt once CG needs more
        than `refresh` iterations, and CG restarts from its current iterate.
        """
        cp.cuda.get_current_stream().synchronize()
        x, r, z, p, Ap = (self.cg[k] for k in ("x", "r", "z", "p", "Ap"))
        total = 0
        with self.stream:
            b = cp.asarray(rhs, dtype=cp.float64) * self.mask
            x[...] = 0.0 if x0 is None else x0 * self.mask
            tol = max(rtol * float(cp.linalg.norm(b)), atol) ** 2
            while True:
                r[...] = b - self.operator.apply(x, Ap)
                if float(r @ r) <= tol:
                    break
                self._precondition(r, z)
                p[...] = z
                self._dot(r, z, self.scalars[0])
                if self.graph is None:
                    self.stream.begin_capture()
                    self._iteration()
                    self.graph = self.stream.end_capture()
                limit = refresh if self.stale else maxiter - total
                for it in range(1, limit + 1):
                    self.graph.launch(self.stream)
                    if float(self.scalars[2]) <= tol:
                        break
                total += it
                if float(self.scalars[2]) <= tol or total >= maxiter:
                    break
                self.stale = False
                self._setup()
            u = x.copy()
        self.stream.synchronize()
        return u, total

    def apply_full(self, u):
        """unconstrained K u in float64, for Dirichlet lifting and reaction forces."""
        u = cp.asarray(u, dtype=cp.float64)  # on the current stream, no race with u
        return self.operator.apply(u, cp.empty(self.ndof), full=True)

    def gather(self, values):
        """global vector from element vectors (n_elems, n), summed in a fixed order."""
        fine = self.levels[0]
        out = cp.empty(self.ndof)
        _gather_rows(
            fine.adj,
            cp.ascontiguousarray(values, dtype=cp.float64).ravel(),
            fine.adj.shape[1],
            out,
        )
        return out

    def load(self, grid, W):
        """global vector sum_e sum_s grid[e, s] W[s] from a voxel grid and W (S, n)."""
        return self.gather(self.to_elements(grid) @ cp.asarray(W))

    def voxel_values(self, u, W):
        """voxel grid of u_e . W[s], e.g. sub-voxel means with W from `mean`."""
        fine = self.levels[0]
        values = cp.asarray(u)[fine.efts] @ cp.asarray(W).T
        return elements_to_grid(values, fine.nel, self.sub)

    def element_energy(self, u):
        """(T, *nvoxels) u_e^T K_t[s] u_e, the unit-coefficient energy of every family."""
        fine = self.levels[0]
        u_e = cp.asarray(u)[fine.efts]
        TS, n = self.K_locals.shape[:2]
        flat = self.K_locals.transpose(2, 0, 1).reshape(n, -1)
        energy = cp.einsum("esi,ei->es", (u_e @ flat).reshape(-1, TS, n), u_e)
        energy = energy.reshape(len(u_e), self.families, -1).transpose(1, 0, 2)
        return cp.stack([elements_to_grid(e, fine.nel, self.sub) for e in energy])


class PointwiseProblem:
    """internal forces and element tangents of a pointwise material on the voxel grid.

    Elements are processed in chunks, so the pointwise tangents never exist for the
    whole grid at once; only the element matrices (n_elems, n, n) are stored.
    """

    def __init__(self, solver, tables, material, params, chunk=2**15, dtype=cp.float32):
        self.solver, self.tables, self.material = solver, tables, material
        self.chunk, self.dtype = (
            chunk,
            dtype,
        )  # tangents only steer the Newton iteration
        self.G = tables.gradient(solver.D)
        fine = solver.levels[0]
        params = cp.asarray(params, dtype=cp.float64)
        self.params = cp.stack(
            [
                grid_to_elements(params[..., k], fine.nel, solver.sub)
                for k in range(params.shape[-1])
            ],
            axis=-1,
        )

    def evaluate(self, u, tangent=True):
        """global internal force vector and, if tangent, element matrices."""
        efts = self.solver.levels[0].efts
        forces = cp.empty(efts.shape)
        K = cp.empty((*efts.shape, efts.shape[1]), self.dtype) if tangent else None
        for start in range(0, len(efts), self.chunk):
            ids = slice(start, start + self.chunk)
            H = self.tables.at_points(self.G, cp.asarray(u)[efts[ids]])
            P, A = self.material(H, self.params[ids])
            forces[ids] = self.tables.forces(self.G, P)
            if tangent:
                K[ids] = self.tables.tangents(self.G, A.astype(self.dtype))
        return self.solver.gather(forces), K
