import cupy as cp
import numpy as np
from cupyx.scipy import sparse

CORNERS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # local corner a = 2 * ax + ay
EDGES = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])  # x = 0, x = 1, y = 0, y = 1

# one thread per row, no atomics: reproducible sums
KERNELS = cp.RawModule(
    code=r"""
extern "C" __global__ void csr_apply(const int* ip, const int* ix, const double* a,
                                     const double* x, double* y, const int n,
                                     const int accumulate)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double s = 0;
    for (int k = ip[i]; k < ip[i + 1]; ++k) s += a[k] * x[ix[k]];
    y[i] = accumulate ? y[i] + s : s;
}

// r = b - A x, d = c0 dinv r
extern "C" __global__ void csr_residual(const int* ip, const int* ix, const double* a,
                                        const double* b, const double* x,
                                        const double* dinv, double* r, double* d,
                                        const double c0, const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double s = 0;
    for (int k = ip[i]; k < ip[i + 1]; ++k) s += a[k] * x[ix[k]];
    r[i] = b[i] - s;
    d[i] = c0 * dinv[i] * r[i];
}

// r -= A d_in, d_out = c1 d_in + c2 dinv r, x += d_out (+ d_in)
extern "C" __global__ void csr_chebyshev(const int* ip, const int* ix, const double* a,
                                         const double* dinv, const double* d_in,
                                         double* d_out, double* r, double* x,
                                         const double c1, const double c2,
                                         const int add_in, const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double s = 0;
    for (int k = ip[i]; k < ip[i + 1]; ++k) s += a[k] * d_in[ix[k]];
    const double ri = r[i] - s;
    r[i] = ri;
    const double dn = c1 * d_in[i] + c2 * dinv[i] * ri;
    d_out[i] = dn;
    x[i] += add_in ? dn + d_in[i] : dn;
}

// r -= A d
extern "C" __global__ void csr_update_residual(const int* ip, const int* ix,
                                               const double* a, const double* d,
                                               double* r, const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double s = 0;
    for (int k = ip[i]; k < ip[i + 1]; ++k) s += a[k] * d[ix[k]];
    r[i] -= s;
}

extern "C" __global__ void dense_matvec(const double* A, const double* x, double* y,
                                        const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double s = 0;
    for (int j = 0; j < n; ++j) s += A[(long long)i * n + j] * x[j];
    y[i] = s;
}

extern "C" __global__ void dot_partial(const double* a, const double* b,
                                       double* partial, const long long n)
{
    __shared__ double cache[256];
    double s = 0;
    for (long long k = blockIdx.x * 256 + threadIdx.x; k < n; k += 256LL * gridDim.x)
        s += a[k] * b[k];
    cache[threadIdx.x] = s;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) cache[threadIdx.x] += cache[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = cache[0];
}

extern "C" __global__ void dot_final(const double* partial, double* out,
                                     const int blocks)
{
    double s = 0;
    for (int k = 0; k < blocks; ++k) s += partial[k];
    *out = s;
}
"""
)
DOT_BLOCKS = 128
_pre_first = cp.ElementwiseKernel(
    "T b, T dinv, float64 c0",
    "T r, T d, T x",
    "r = b; d = c0 * dinv * b; x = d",
    "pre_first",
)
_add = cp.ElementwiseKernel("T a", "T x", "x += a", "add")
_xr_update = cp.ElementwiseKernel(
    "T p, T Ap, raw T s",
    "T x, T r",
    "T alpha = s[0] / s[1]; x += alpha * p; r -= alpha * Ap",
    "xr_update",
)
_p_update = cp.ElementwiseKernel(
    "T z, raw T s", "T p", "p = z + (s[4] - s[3]) / s[0] * p", "p_update"
)


def launch(function, threads, args):
    function(((threads + 255) // 256,), (256,), args)


def isotropic_tensor(D, lam, mu):
    """(D * D, D * D) isotropic elasticity tensor."""
    eye = np.eye(D)
    C = lam * np.einsum("ij,kl->ijkl", eye, eye)
    C += mu * (np.einsum("ik,jl->ijkl", eye, eye) + np.einsum("il,jk->ijkl", eye, eye))
    return C.reshape(D * D, D * D)


class UnitSquare:
    """bilinear shape functions at the Gauss points of the unit square.

    Every leaf of the quadtree is a scaled copy of it, so these tables hold the whole
    geometry: a family integrated here scales with h^p on a leaf of size h (p = 0 for
    gradient-gradient forms in 2D, p = 2 for mass-type forms). Element dofs are
    ordered by field, then by corner.
    """

    def __init__(self, order=2):
        gauss, weights = np.polynomial.legendre.leggauss(order)
        t = 0.5 * (gauss + 1.0)
        x, y = [a.ravel()[:, None] for a in np.meshgrid(t, t, indexing="ij")]
        fx = np.where(CORNERS[:, 0], x, 1.0 - x)
        fy = np.where(CORNERS[:, 1], y, 1.0 - y)
        dfx = np.where(CORNERS[:, 0], 1.0, -1.0)
        dfy = np.where(CORNERS[:, 1], 1.0, -1.0)
        self.N = fx * fy  # (Q, 4)
        self.dN = np.stack([dfx * fy, fx * dfy], -1)  # (Q, 4, 2)
        self.w = np.outer(weights, weights).ravel() / 4.0
        self.D, self.n = 2, 4

    def gradient(self, nfields):
        """(Q, nfields * D, nfields * n): element dofs -> du_i / dx_j at i * D + j."""
        return np.einsum("fg,qaj->qfjga", np.eye(nfields), self.dN).reshape(
            len(self.w), nfields * self.D, nfields * self.n
        )

    def value(self, nfields):
        """(Q, nfields, nfields * n): element dofs -> u_i."""
        return np.einsum("fg,qa->qfga", np.eye(nfields), self.N).reshape(
            len(self.w), nfields, nfields * self.n
        )

    def integrate(self, op, C=None):
        """(n, n) matrix of the quadratic form op^T C op."""
        Cop = op if C is None else np.einsum("ij,qjk->qik", C, op)
        return np.einsum("q,qia,qib->ab", self.w, op, Cop)

    def mean(self, op):
        """element average of a pointwise operator (Q, ...)."""
        return np.einsum("q,q...->...", self.w, op)


class QuadTree:
    """2:1 balanced quadtree of square leaves (level, i, j) on a base grid.

    Leaf corners live on the integer lattice of the deepest allowed level `depth`.
    """

    def __init__(self, base, lengths, depth):
        self.base, self.depth = list(base), depth
        self.lengths = np.asarray(lengths, dtype=float)
        self.spacing = self.lengths[0] / base[0] / 2**depth  # lattice spacing
        assert np.isclose(self.lengths[1] / base[1] / 2**depth, self.spacing)
        i, j = np.meshgrid(*[np.arange(n) for n in base], indexing="ij")
        self.level = np.zeros(i.size, dtype=np.int64)
        self.ij = np.stack([i.ravel(), j.ravel()], 1)

    @property
    def size(self):  # leaf edge in lattice units
        return 2 ** (self.depth - self.level)

    @property
    def h(self):
        return self.size * self.spacing

    @property
    def centers(self):
        return (self.ij + 0.5) * self.h[:, None]

    def _keys(self, level, ij):
        return (level * (self.base[0] << self.depth) + ij[:, 0]) * (
            self.base[1] << self.depth
        ) + ij[:, 1]

    def _set(self, level, ij, mask):
        children = (2 * ij[mask][:, None, :] + CORNERS).reshape(-1, 2)
        level = np.concatenate([level[~mask], np.repeat(level[mask] + 1, 4)])
        ij = np.concatenate([ij[~mask], children])
        order = np.lexsort(
            (ij[:, 1] * self.size_of(level), ij[:, 0] * self.size_of(level))
        )
        self.level, self.ij = level[order], ij[order]

    def size_of(self, level):
        return 2 ** (self.depth - level)

    def refine(self, mask):
        """split the marked leaves, then split more until the tree is 2:1 balanced."""
        mask = np.asarray(mask, dtype=bool) & (self.level < self.depth)
        while mask.any():
            self._set(self.level, self.ij, mask)
            mask = self._unbalanced()

    def _unbalanced(self):  # leaves with an edge neighbour two or more levels finer
        keys = self._keys(self.level, self.ij)
        order = np.argsort(keys)
        ordered = keys[order]
        mask = np.zeros(len(self.level), dtype=bool)
        limits = np.array(self.base)[None, :] << self.level[:, None]
        for step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = self.ij + np.array(step)
            inside = ((neighbour >= 0) & (neighbour < limits)).all(1)
            for m in range(self.level.max() - 1):
                deep = np.flatnonzero(inside & (self.level >= m + 2))
                shift = (self.level[deep] - m)[:, None]
                key = self._keys(np.full(len(deep), m), neighbour[deep] >> shift)
                pos = np.searchsorted(ordered, key).clip(0, len(ordered) - 1)
                mask[order[pos[ordered[pos] == key]]] = True
        return mask

    def refine_to(self, target):
        """refine until every leaf reaches target(x, y) at its center and corners."""
        while True:
            points = [self.centers] + [(self.ij + c) * self.h[:, None] for c in CORNERS]
            wanted = np.max([target(p[:, 0], p[:, 1]) for p in points], axis=0)
            mask = self.level < np.minimum(wanted, self.depth)
            if not mask.any():
                return self
            self.refine(mask)

    def truncate(self, level):
        """tree with every leaf deeper than level replaced by its ancestor."""
        tree = QuadTree.__new__(QuadTree)
        tree.__dict__.update(self.__dict__)
        new = np.minimum(self.level, level)
        ij = self.ij >> (self.level - new)[:, None]
        _, first = np.unique(self._keys(new, ij), return_index=True)
        tree.level, tree.ij = new[first], ij[first]
        order = np.lexsort((ij[first, 1] * tree.size, ij[first, 0] * tree.size))
        tree.level, tree.ij = tree.level[order], tree.ij[order]
        return tree

    def find(self, level, ij):
        """leaf containing each cell (level, ij), -1 where the cell is subdivided."""
        keys = self._keys(self.level, self.ij)
        order = np.argsort(keys)
        out = np.full(len(level), -1)
        for m in range(int(self.level.max()) + 1):
            inside = level >= m
            key = self._keys(
                np.full(len(level), m), ij >> np.maximum(level - m, 0)[:, None]
            )
            pos = np.searchsorted(keys[order], key).clip(0, len(keys) - 1)
            found = inside & (keys[order][pos] == key)
            out[found] = order[pos[found]]
        return out

    def pixels(self):
        """(nx, ny) leaf index of every lattice cell at the deepest leaf level."""
        top = int(self.level.max())
        n = [b << top for b in self.base]
        pix = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1)
        return self.find(np.full(pix[..., 0].size, top), pix.reshape(-1, 2)).reshape(n)

    def copy(self):
        tree = QuadTree.__new__(QuadTree)
        tree.__dict__.update(self.__dict__)
        return tree


class Space:
    """continuous bilinear space with hanging-node constraints on a quadtree.

    Element vectors follow from the free dofs through u_e[a] = sum_k w[e, a, k]
    u[idx[e, a, k]]; hanging corners interpolate their constraining nodes, chained
    through as many levels as needed. Global dofs are ordered by field, then node.
    """

    def __init__(self, tree, nfields, fixed):
        self.tree, self.nfields = tree.copy(), nfields  # refinement mutates tree
        size = tree.size[:, None, None]
        corners = (tree.ij[:, None, :] + CORNERS) * size  # (E, 4, 2) lattice
        stride = (tree.base[1] << tree.depth) + 1
        keys = corners[..., 0] * stride + corners[..., 1]
        nodes, corner_nodes = np.unique(keys, return_inverse=True)
        corner_nodes = corner_nodes.reshape(-1, 4)
        nn = len(nodes)

        mids = (corners[:, EDGES[:, 0]] + corners[:, EDGES[:, 1]]) // 2
        mid_keys = mids[..., 0] * stride + mids[..., 1]
        pos = np.searchsorted(nodes, mid_keys).clip(0, nn - 1)
        hanging = (nodes[pos] == mid_keys) & (size[:, :, 0] >= 2)
        rows = pos[hanging]
        masters = corner_nodes[:, EDGES][hanging]  # (H, 2)
        is_hanging = np.zeros(nn, dtype=bool)
        is_hanging[rows] = True
        free_rows = np.flatnonzero(~is_hanging)
        C0 = sparse.csr_matrix(
            (
                cp.asarray(
                    np.concatenate(
                        [np.ones(len(free_rows)), np.full(2 * len(rows), 0.5)]
                    )
                ),
                (
                    cp.asarray(np.concatenate([free_rows, np.repeat(rows, 2)])),
                    cp.asarray(np.concatenate([free_rows, masters.ravel()])),
                ),
            ),
            shape=(nn, nn),
        )
        C = C0
        hanging_gpu = cp.asarray(is_hanging)
        while bool(hanging_gpu[C.indices].any()):
            C = C @ C0
        C.sum_duplicates()
        free = np.cumsum(~is_hanging) - 1
        C = sparse.csr_matrix(
            (C.data, cp.asarray(free)[C.indices], C.indptr), shape=(nn, len(free_rows))
        )
        self.constraints = C  # all nodes -> free nodes
        self.nfree = len(free_rows)
        self.ndof = nfields * self.nfree
        self.corner_nodes = corner_nodes
        self.nodes, self.free_nodes = nodes, free_rows

        counts = cp.diff(C.indptr).get()
        K = int(counts.max())
        slot = np.arange(K)[None, :]
        starts = C.indptr[:-1].get()[:, None] + slot
        valid = slot < counts[:, None]
        starts = np.where(valid, starts, 0)
        node_idx = np.where(valid, C.indices.get()[starts], 0)
        node_w = np.where(valid, C.data.get()[starts], 0.0)
        idx = node_idx[corner_nodes]  # (E, 4, K)
        w = node_w[corner_nodes]
        offsets = (np.arange(nfields) * self.nfree)[None, :, None, None]
        self.idx = cp.asarray((idx[:, None] + offsets).reshape(len(idx), -1, K))
        self.w = cp.asarray(
            np.broadcast_to(w[:, None], (len(w), nfields, 4, K)).reshape(len(w), -1, K)
        )
        self.n = 4 * nfields

        xy = np.stack([nodes[free_rows] // stride, nodes[free_rows] % stride], 1)
        self.xy = xy * tree.spacing  # (nfree, 2)
        self.fixed = np.asarray(fixed(self.xy)).reshape(nfields, -1).ravel()
        self.mask = cp.asarray(~self.fixed, dtype=cp.float64)

        E = len(tree.level)
        rows = self.idx.ravel()
        cols = cp.repeat(cp.arange(E * self.n), K)
        keep = self.w.ravel() != 0
        self.gatherer = sparse.csr_matrix(
            (self.w.ravel()[keep], (rows[keep], cols[keep])),
            shape=(self.ndof, E * self.n),
        )
        self.localizer = self.gatherer.T.tocsr()

    def local(self, u):
        """(E, n) element vectors of a global vector u."""
        return (self.localizer @ cp.asarray(u, dtype=cp.float64)).reshape(-1, self.n)

    def gather(self, values):
        """global vector from element vectors (E, n), summed in a fixed order."""
        return self.gatherer @ cp.asarray(values, dtype=cp.float64).ravel()

    def interpolation(self, old):
        """(ndof, old.ndof) interpolation of a space on a coarser tree, exact if nested.

        Every leaf of this tree must lie inside a leaf of old.tree, as for a truncated
        tree (multigrid levels) or the tree before refinement (field transfer).
        """
        tree, otree = self.tree, old.tree
        # one element corner per node, then the old leaf containing that element
        first = np.full(len(self.nodes), -1)
        flat = self.corner_nodes.ravel()
        first[flat[::-1]] = np.arange(flat.size)[::-1]
        e, a = np.divmod(first, 4)
        oe = otree.find(tree.level[e], tree.ij[e])
        assert (oe >= 0).all()
        lattice = (tree.ij[e] + CORNERS[a]) * tree.size[e][:, None]
        size = otree.size[oe][:, None]
        local = (lattice - otree.ij[oe] * size) / size
        weights = np.prod(
            np.where(CORNERS[None], local[:, None, :], 1.0 - local[:, None, :]), -1
        )  # (nn, 4)
        B = sparse.csr_matrix(
            (
                cp.asarray(weights.ravel()),
                (
                    cp.asarray(np.repeat(np.arange(len(e)), 4)),
                    cp.asarray(old.corner_nodes[oe].ravel()),
                ),
            ),
            shape=(len(e), len(old.nodes)),
        )
        P = (B @ old.constraints)[cp.asarray(self.free_nodes)]
        P = sparse.kron(sparse.identity(self.nfields, format="csr"), P, format="csr")
        P.eliminate_zeros()
        return P.tocsr()

    def prolongation(self, coarse):
        """interpolation from a truncated tree, zero on the fixed dofs of both."""
        P = (
            sparse.diags(self.mask)
            @ self.interpolation(coarse)
            @ sparse.diags(coarse.mask)
        )
        P.eliminate_zeros()
        return P.tocsr()


class Multigrid:
    """flexible PCG with a Galerkin V-cycle on the truncations of a quadtree.

    The operator is sum_t sum_e c[t, e] h_e^p_t T_e^T K_t T_e with the unit-square
    families K_t. Its CSR values are a fixed sparse map M of the coefficients, built
    once per mesh, so any coefficient update is one sparse product. Coarse levels are
    the spaces on the tree truncated one level at a time, with R A P products.
    """

    def __init__(
        self,
        tree,
        families,
        nfields,
        fixed,
        smoothing=(2, 4, 8, 16),
        coarse_dofs=500,
        eig_ratio=30.0,
        power_iters=8,
    ):
        self.spaces = [Space(tree, nfields, fixed)]
        level = int(tree.level.max())
        while self.spaces[-1].ndof > coarse_dofs and level > 0:
            level -= 1
            self.spaces.append(Space(tree.truncate(level), nfields, fixed))
        self.space = self.spaces[0]
        self.ndof, self.mask = self.space.ndof, self.space.mask
        self.P = [f.prolongation(c) for f, c in zip(self.spaces[:-1], self.spaces[1:])]
        self.R = [P.T.tocsr() for P in self.P]
        self.smoothing = [
            smoothing[min(i, len(smoothing) - 1)] for i in range(len(self.P))
        ]
        self.eig_ratio, self.power_iters = eig_ratio, power_iters
        self.power = [None] * len(self.P)
        self.families = [cp.asarray(K) for K, _ in families]
        self.exponents = [p for _, p in families]
        self.h = cp.asarray(tree.h)
        self._build_map()
        self.levels = None
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.scalars = cp.zeros(5)
        self.partial = cp.zeros(DOT_BLOCKS)

    def _build_map(self, chunk=4096):
        space = self.space
        E = space.idx.shape[0]
        a, b = np.nonzero(np.any([k.get() != 0 for k in self.families], axis=0))
        rows, cols, weight, elem, pair = [], [], [], [], []
        for start in range(0, E, chunk):
            idx, w = space.idx[start : start + chunk], space.w[start : start + chunk]
            ww = w[:, a, :, None] * w[:, b, None, :]
            keep = ww != 0
            shape = ww.shape
            rows.append(cp.broadcast_to(idx[:, a, :, None], shape)[keep])
            cols.append(cp.broadcast_to(idx[:, b, None, :], shape)[keep])
            weight.append(ww[keep])
            e = cp.arange(start, start + len(idx))[:, None, None, None]
            elem.append(cp.broadcast_to(e, shape)[keep])
            pair.append(
                cp.broadcast_to(cp.arange(len(a))[None, :, None, None], shape)[keep]
            )
        rows, cols, weight, elem, pair = map(
            cp.concatenate, (rows, cols, weight, elem, pair)
        )
        keys, entry = cp.unique(rows * self.ndof + cols, return_inverse=True)
        entry = entry.ravel()
        self.indices = (keys % self.ndof).astype(cp.int32)
        self.indptr = cp.searchsorted(
            keys // self.ndof, cp.arange(self.ndof + 1)
        ).astype(cp.int32)
        data, mrows, mcols = [], [], []
        for t, (Kt, p) in enumerate(zip(self.families, self.exponents)):
            values = (
                weight
                * Kt[cp.asarray(a)[pair], cp.asarray(b)[pair]]
                * self.h[elem] ** p
            )
            nz = values != 0
            data.append(values[nz])
            mrows.append(entry[nz])
            mcols.append(elem[nz] + t * E)
        self.M = sparse.csr_matrix(
            (cp.concatenate(data), (cp.concatenate(mrows), cp.concatenate(mcols))),
            shape=(len(keys), len(self.families) * E),
        )
        fixed = cp.asarray(self.space.fixed)
        row = cp.searchsorted(self.indptr, cp.arange(len(keys)), side="right") - 1
        self.keep = cp.asarray(~(fixed[row] | fixed[self.indices]), dtype=cp.float64)
        self.fixed_diag = cp.asarray(
            fixed[row] & (row == self.indices), dtype=cp.float64
        )

    def update(self, coeffs, hierarchy=True):
        """new coefficients, one (E,) array per family; hierarchy=False keeps the V-cycle."""
        c = cp.concatenate([cp.asarray(ci, dtype=cp.float64).ravel() for ci in coeffs])
        data = self.M @ c
        if self.levels is None:
            shape = (self.ndof, self.ndof)
            self.A_full = sparse.csr_matrix(
                (data, self.indices, self.indptr), shape=shape
            )
            self.A = sparse.csr_matrix(
                (data.copy(), self.indices, self.indptr), shape=shape
            )
        self.A_full.data[...] = data  # in place, the captured graph holds the pointers
        self.A.data[...] = data * self.keep + self.fixed_diag
        self.stale = not hierarchy
        if hierarchy or self.levels is None:
            self.setup()

    def setup(self):
        """V-cycle hierarchy from the current operator."""
        levels = [self.A.copy()]
        for R, P, space in zip(self.R, self.P, self.spaces[1:]):
            coarse = R @ levels[-1] @ P
            coarse = coarse + sparse.diags(cp.asarray(space.fixed, dtype=cp.float64))
            levels.append(coarse.tocsr())
        self.levels = []
        for i, A in enumerate(levels):
            n = A.shape[0]
            lv = {"A": A, "n": n}
            lv.update({k: cp.zeros(n) for k in ("b", "x", "r", "d", "d2")})
            if i < len(self.P):
                lv["dinv"] = 1.0 / A.diagonal()
                lv["cheb"] = self._chebyshev(i, A, lv["dinv"])
            self.levels.append(lv)
        self.coarse_inverse = cp.linalg.inv(levels[-1].toarray())
        self.x, self.p, self.Ap = (cp.zeros(self.ndof) for _ in range(3))
        self.graph = None

    def _chebyshev(self, i, A, dinv):  # step coefficients on [lmax / ratio, lmax]
        iters = self.power_iters
        if self.power[i] is None:
            self.power[i] = cp.random.default_rng(0).random(A.shape[0])
            iters *= 3
        v = self.power[i]
        for _ in range(iters):
            v = dinv * (A @ (v / cp.linalg.norm(v)))
        self.power[i] = v
        upper = 1.05 * float(cp.linalg.norm(v))
        lower = upper / self.eig_ratio
        theta, delta = 0.5 * (upper + lower), 0.5 * (upper - lower)
        sigma = theta / delta
        rho = 1.0 / sigma
        steps = [1.0 / theta]
        for _ in range(self.smoothing[i] - 1):
            rho_next = 1.0 / (2.0 * sigma - rho)
            steps.append((rho_next * rho, 2.0 * rho_next / delta))
            rho = rho_next
        return steps

    def _csr(self, A, x, y, accumulate=False):  # y = A x, or y += A x
        launch(
            KERNELS.get_function("csr_apply"),
            A.shape[0],
            (
                A.indptr,
                A.indices,
                A.data,
                x,
                y,
                np.int32(A.shape[0]),
                np.int32(accumulate),
            ),
        )

    def _smooth(self, lv, pre):  # Chebyshev-Jacobi, pre-smoothing starts from x = 0
        A, cheb = lv["A"], lv["cheb"]
        d, d_next = lv["d"], lv["d2"]
        n = np.int32(lv["n"])
        if pre:
            _pre_first(lv["b"], lv["dinv"], cheb[0], lv["r"], d, lv["x"])
        else:
            launch(
                KERNELS.get_function("csr_residual"),
                lv["n"],
                (
                    A.indptr,
                    A.indices,
                    A.data,
                    lv["b"],
                    lv["x"],
                    lv["dinv"],
                    lv["r"],
                    d,
                    cheb[0],
                    n,
                ),
            )
            if len(cheb) == 1:
                _add(d, lv["x"])
        for k, (c1, c2) in enumerate(cheb[1:]):
            launch(
                KERNELS.get_function("csr_chebyshev"),
                lv["n"],
                (
                    A.indptr,
                    A.indices,
                    A.data,
                    lv["dinv"],
                    d,
                    d_next,
                    lv["r"],
                    lv["x"],
                    c1,
                    c2,
                    np.int32(not pre and k == 0),
                    n,
                ),
            )
            d, d_next = d_next, d
        if pre:
            launch(
                KERNELS.get_function("csr_update_residual"),
                lv["n"],
                (A.indptr, A.indices, A.data, d, lv["r"], n),
            )

    def _vcycle(self):  # levels[0]["b"] -> levels[0]["x"]
        L = len(self.P)
        for i in range(L):
            self._smooth(self.levels[i], pre=True)
            self._csr(self.R[i], self.levels[i]["r"], self.levels[i + 1]["b"])
        coarse = self.levels[L]
        launch(
            KERNELS.get_function("dense_matvec"),
            coarse["n"],
            (self.coarse_inverse, coarse["b"], coarse["x"], np.int32(coarse["n"])),
        )
        for i in reversed(range(L)):
            self._csr(self.P[i], self.levels[i + 1]["x"], self.levels[i]["x"], True)
            self._smooth(self.levels[i], pre=False)

    def _dot(self, a, b, out):  # deterministic two-stage reduction
        KERNELS.get_function("dot_partial")(
            (DOT_BLOCKS,), (256,), (a, b, self.partial, np.int64(a.size))
        )
        KERNELS.get_function("dot_final")(
            (1,), (1,), (self.partial, out, np.int32(DOT_BLOCKS))
        )

    def _iteration(self):  # one flexible PCG step on the persistent state
        s, fine = self.scalars, self.levels[0]
        x, r, z, p, Ap = self.x, fine["b"], fine["x"], self.p, self.Ap
        self._csr(self.A, p, Ap)
        self._dot(p, Ap, s[1])
        _xr_update(p, Ap, s, x, r)
        self._dot(r, r, s[2])
        self._dot(r, z, s[3])
        self._vcycle()
        self._dot(r, z, s[4])
        _p_update(z, s, p)  # flexible beta, the V-cycle may be stale
        s[0:1] = s[4:5]

    def vcycle(self, r):
        """one V-cycle z = B r (for testing)."""
        self.levels[0]["b"][...] = r * self.mask
        self._vcycle()
        return self.levels[0]["x"].copy()

    def solve(self, rhs, x0=None, rtol=1e-8, maxiter=500, refresh=20):
        """flexible PCG; returns (u, iterations), zero on the fixed dofs.

        A V-cycle kept stale by `update(..., hierarchy=False)` is rebuilt once CG needs
        more than `refresh` iterations, and CG restarts from its current iterate.
        """
        cp.cuda.get_current_stream().synchronize()
        total = 0
        with self.stream:
            b = cp.asarray(rhs, dtype=cp.float64) * self.mask
            tol = (rtol * float(cp.linalg.norm(b))) ** 2
            x0 = cp.zeros(self.ndof) if x0 is None else x0 * self.mask
            self.x[...] = x0
            while True:
                fine = self.levels[0]
                r, z = fine["b"], fine["x"]
                self._csr(self.A, self.x, r)
                r[...] = b - r
                if float(r @ r) <= tol:
                    break
                self._vcycle()
                self.p[...] = z
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
                if float(self.scalars[2]) <= tol or total >= maxiter or not self.stale:
                    break
                self.stale = False
                self.setup()
            u = self.x.copy()
        self.stream.synchronize()
        return u, total

    def apply_full(self, u):
        """unconstrained K u, for Dirichlet lifting and reaction forces."""
        u = cp.asarray(u, dtype=cp.float64)
        out = cp.empty(self.ndof)
        self._csr(self.A_full, u, out)
        return out

    def element_values(self, u, W):
        """(E,) element values u_e . W, e.g. element means."""
        return self.space.local(u) @ cp.asarray(W)

    def element_energy(self, u):
        """(T, E) u_e^T K_t u_e h^p_t, the unit-coefficient energy of every family."""
        u_e = self.space.local(u)
        return cp.stack(
            [
                cp.einsum("ea,ea->e", u_e @ Kt, u_e) * self.h**p
                for Kt, p in zip(self.families, self.exponents)
            ]
        )

    def load(self, values, W, p):
        """global vector sum_e values[e] h_e^p W."""
        return self.space.gather(
            (cp.asarray(values) * self.h**p)[:, None] * cp.asarray(W)[None, :]
        )
