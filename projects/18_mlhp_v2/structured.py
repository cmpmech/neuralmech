"""matrix-free multigrid-preconditioned CG for 2D voxel elasticity on a structured grid."""

import cupy as cp
import numpy as np

# one thread per node computes both displacement rows from its four elements; dofs are
# (field, i, j) with j fastest, so neighbours are found by index arithmetic, no tables
KERNELS = r"""
#define LEVEL_ARGS const int ex, const int ey, const int stride, const float* coeff, \
                   const T* mats, const unsigned char* mask
#define LEVEL ex, ey, stride, coeff, mats, mask

template <typename T>
__device__ __forceinline__ T dot8(const T* a, const T* b)
{
    T s = 0;
    #pragma unroll
    for (int q = 0; q < 8; ++q) s += a[q] * b[q];
    return s;
}

// CH = 0: stored element matrices; CH > 0: CH^2 voxel coefficients per element;
// CH = -1: K_e = sum_p m_p B_p from six material moments per element in coeff
template <typename T, typename U, int CH, bool DIAG>
__device__ __forceinline__ void node_rows(LEVEL_ARGS, const U* u, const int i,
                                          const int j, T acc[2])
{
    const int ny = ey + 1;
    const long long nn = (long long)(ex + 1) * ny;
    acc[0] = 0;
    acc[1] = 0;
    #pragma unroll
    for (int a = 0; a < 2; ++a) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int ei = i - 1 + a, ej = j - 1 + b;
            if (ei < 0 || ej < 0 || ei >= ex || ej >= ey) continue;
            T ue[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (!DIAG) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const long long n = (long long)(ei + (m & 1)) * ny + ej + (m >> 1);
                    ue[2 * m] = (T)u[n];
                    ue[2 * m + 1] = (T)u[nn + n];
                }
            }
            const int l = (1 - a) + 2 * (1 - b);
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int r = 2 * l + f;
                if (CH == 0) {
                    const T* Kr = mats + ((long long)ei * ey + ej) * 64 + r * 8;
                    acc[f] += DIAG ? Kr[r] : dot8(Kr, ue);
                } else if (CH < 0) {
                    const float* m = coeff + ((long long)ei * ey + ej) * 6;
                    #pragma unroll
                    for (int q = 0; q < 6; ++q) {
                        const T* Kr = mats + q * 64 + r * 8;
                        acc[f] += (T)m[q] * (DIAG ? Kr[r] : dot8(Kr, ue));
                    }
                } else {
                    #pragma unroll
                    for (int ka = 0; ka < (CH > 0 ? CH : 1); ++ka) {
                        #pragma unroll
                        for (int kb = 0; kb < (CH > 0 ? CH : 1); ++kb) {
                            const T c = (T)coeff[(long long)(ei * CH + ka) * stride
                                                 + ej * CH + kb];
                            const T* Kr = mats + (ka * CH + kb) * 64 + r * 8;
                            acc[f] += c * (DIAG ? Kr[r] : dot8(Kr, ue));
                        }
                    }
                }
            }
        }
    }
}

#define NODE_KERNEL(name, args, body)                                              \
template <typename T, int CH>                                                      \
__global__ void name(LEVEL_ARGS, args)                                             \
{                                                                                  \
    const int ny = ey + 1;                                                         \
    const long long nn = (long long)(ex + 1) * ny;                                 \
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;          \
    if (t >= nn) return;                                                           \
    const int i = t / ny, j = t % ny;                                              \
    T acc[2];                                                                      \
    body                                                                           \
}
#define ROWS(u) node_rows<T, T, CH, false>(LEVEL, u, i, j, acc)
#define FOR_FIELDS(stmt) for (int f = 0; f < 2; ++f) { const long long k = f * nn + t; stmt }
#define AU(u) (mask[k] ? acc[f] : u[k])
#define COMMA ,

// out = A u, fixed dofs act as identity rows
NODE_KERNEL(apply, const T* u COMMA T* out, ROWS(u); FOR_FIELDS(out[k] = AU(u);))

NODE_KERNEL(diagonal, T* out,
            node_rows<T COMMA T COMMA CH COMMA true>(LEVEL, (const T*)0, i, j, acc);
            FOR_FIELDS(out[k] = mask[k] ? acc[f] : (T)1;))

// r = b - A x, d = dinv r / theta
NODE_KERNEL(residual,
            const T* b COMMA const T* x COMMA const T* dinv COMMA const T* cheb
            COMMA T* r COMMA T* d,
            ROWS(x); FOR_FIELDS(r[k] = b[k] - AU(x); d[k] = dinv[k] * r[k] * cheb[0];))

// r -= A d
NODE_KERNEL(update_residual, const T* d COMMA T* r, ROWS(d); FOR_FIELDS(r[k] -= AU(d);))

// Chebyshev step: r -= A d_in, d_out = c1 d_in + c2 dinv r, x += d_out (+ d_in)
NODE_KERNEL(chebyshev,
            const T* dinv COMMA const T* cheb COMMA const T* d_in COMMA T* d_out
            COMMA T* r COMMA T* x COMMA const int add_in,
            ROWS(d_in);
            FOR_FIELDS(const T ri = r[k] - AU(d_in); r[k] = ri;
                       const T dn = cheb[0] * d_in[k] + cheb[1] * dinv[k] * ri;
                       d_out[k] = dn; x[k] += add_in ? dn + d_in[k] : dn;))

// out = A u of the outer CG: float64 arithmetic, u and out stored as U and V
template <typename U, typename V, int CH>
__global__ void apply64(const int ex, const int ey, const int stride, const float* coeff,
                        const double* mats, const unsigned char* mask, const U* u,
                        V* out)
{
    const int ny = ey + 1;
    const long long nn = (long long)(ex + 1) * ny;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nn) return;
    const int i = t / ny, j = t % ny;
    double acc[2];
    node_rows<double, U, CH, false>(LEVEL, u, i, j, acc);
    for (int f = 0; f < 2; ++f) {
        const long long k = f * nn + t;
        out[k] = (V)(mask[k] ? acc[f] : (double)u[k]);
    }
}

// xf += mask (P xc), bilinear interpolation from the coarse nodes; ex, ey are fine
template <typename T>
__global__ void prolong(const int ex, const int ey, const T* xc, T* xf,
                        const unsigned char* mask)
{
    const int ny = ey + 1, nyc = ey / 2 + 1;
    const long long nn = (long long)(ex + 1) * ny, nnc = (long long)(ex / 2 + 1) * nyc;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nn) return;
    const int i = t / ny, j = t % ny;
    const int i0 = i / 2, j0 = j / 2, di = i & 1, dj = j & 1;
    for (int f = 0; f < 2; ++f) {
        const T* c = xc + f * nnc;
        T v = c[(long long)i0 * nyc + j0];
        if (di) v = 0.5 * (v + c[(long long)(i0 + 1) * nyc + j0]);
        if (dj) {
            T w = c[(long long)i0 * nyc + j0 + 1];
            if (di) w = 0.5 * (w + c[(long long)(i0 + 1) * nyc + j0 + 1]);
            v = 0.5 * (v + w);
        }
        const long long k = f * nn + t;
        xf[k] += mask[k] ? v : (T)0;
    }
}

// bc = mask (P^T rf); ex, ey are fine
template <typename T>
__global__ void restrict_(const int ex, const int ey, const T* rf, T* bc,
                          const unsigned char* mask)
{
    const int ny = ey + 1, nyc = ey / 2 + 1;
    const long long nn = (long long)(ex + 1) * ny, nnc = (long long)(ex / 2 + 1) * nyc;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nnc) return;
    const int I = t / nyc, J = t % nyc;
    for (int f = 0; f < 2; ++f) {
        T s = 0;
        for (int di = -1; di <= 1; ++di) {
            const int i = 2 * I + di;
            if (i < 0 || i > ex) continue;
            for (int dj = -1; dj <= 1; ++dj) {
                const int j = 2 * J + dj;
                if (j < 0 || j > ey) continue;
                const T w = (di ? 0.5 : 1.0) * (dj ? 0.5 : 1.0);
                s += w * rf[f * nn + (long long)i * ny + j];
            }
        }
        const long long k = f * nnc + t;
        bc[k] = mask[k] ? s : (T)0;
    }
}

// u_e^T T_k^T K T_k u_e per voxel k of the elements with sub x sub voxels; the
// element translation is removed in float64, which K ignores, so float32 suffices
__global__ void energy(const int ex, const int ey, const int sub, const float* G,
                       const double* u, float* out)
{
    const int ny = ey + 1, vy = ey * sub;
    const long long nn = (long long)(ex + 1) * ny;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= (long long)ex * sub * vy) return;
    const int vi = t / vy, vj = t % vy, ei = vi / sub, ej = vj / sub;
    const float* K = G + ((vi % sub) * sub + vj % sub) * 64;
    const long long n0 = (long long)ei * ny + ej;
    const double u0 = u[n0], v0 = u[nn + n0];
    float ue[8];
    for (int m = 0; m < 4; ++m) {
        const long long n = (long long)(ei + (m & 1)) * ny + ej + (m >> 1);
        ue[2 * m] = (float)(u[n] - u0);
        ue[2 * m + 1] = (float)(u[nn + n] - v0);
    }
    float e = 0;
    for (int r = 0; r < 8; ++r) e += ue[r] * dot8(K + r * 8, ue);
    out[t] = e;
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

template <typename A, typename B>
__global__ void dot_partial(const A* a, const B* b, double* partial, const long long n)
{
    __shared__ double cache[256];
    double acc = 0;
    for (long long k = blockIdx.x * 256 + threadIdx.x; k < n; k += 256LL * gridDim.x)
        acc += (double)a[k] * (double)b[k];
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
    """compiled kernel or template instance, e.g. `kernel("apply<float, 1>")`."""
    if name not in _functions:
        module = cp.RawModule(code=KERNELS, name_expressions=[name])
        _functions[name] = module.get_function(name)
    return _functions[name]


def launch(function, threads, args):
    function(((threads + 255) // 256,), (256,), args)


_pre_first = cp.ElementwiseKernel(
    "T b, T dinv, raw T cheb",
    "T r, T d, T x",
    "r = b; d = dinv * b * cheb[0]; x = d",
    "pre_first",
)
_add = cp.ElementwiseKernel("T a", "T x", "x += a", "add")
_cast = cp.ElementwiseKernel("S a", "T b", "b = (T)a", "cast")
_mask = cp.ElementwiseKernel("uint8 m", "T x", "x = m ? x : (T)0", "mask")
_init_residual = cp.ElementwiseKernel(
    "S b, uint8 m", "T r", "r = m ? (T)b - r : (T)0", "init_residual"
)
_masked_norm2 = cp.ReductionKernel(
    "S b, uint8 m",
    "float64 y",
    "m ? (double)b * b : 0.0",
    "a + b",
    "y = a",
    "0",
    "masked_norm2",
)
# x and r are float64, the search direction p, A p and z are float32
_xr_update = cp.ElementwiseKernel(
    "F p, F Ap, raw D s",
    "D x, D r",
    "D alpha = s[0] / s[1]; x += alpha * p; r -= alpha * Ap",
    "xr_update",
)
_p_update = cp.ElementwiseKernel(
    "F z, raw D s", "F p", "p = z + (s[4] - s[3]) / s[0] * p", "p_update"
)


def element_stiffness(h, nu):
    """(8, 8) plane stress bilinear element matrix for unit Young's modulus.

    Local nodes m = mi + 2 mj at (mi hx, mj hy), local dofs 2 m + field.
    """
    D = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2]])
    D /= 1.0 - nu**2
    K = np.zeros((8, 8))
    gauss = [0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)]
    for x in gauss:
        for y in gauss:
            B = np.zeros((3, 8))
            for m in range(4):
                mi, mj = m & 1, m >> 1
                dx = (1 if mi else -1) * (y if mj else 1 - y) / h[0]
                dy = (x if mi else 1 - x) * (1 if mj else -1) / h[1]
                B[:, 2 * m : 2 * m + 2] = [[dx, 0.0], [0.0, dy], [dy, dx]]
            K += B.T @ D @ B * 0.25 * h[0] * h[1]
    return K


def moment_weights(sub):
    """(sub^2, 6) integrals of 1, x, y, x^2, y^2, xy over each voxel of a unit element."""
    edges = np.arange(sub + 1) / sub
    a, b = edges[:-1], edges[1:]
    integrals = [b - a, (b**2 - a**2) / 2, (b**3 - a**3) / 3]
    X, Y = [np.repeat(f, sub) for f in integrals], [np.tile(f, sub) for f in integrals]
    return np.stack(
        [X[0] * Y[0], X[1] * Y[0], X[0] * Y[1], X[2] * Y[0], X[0] * Y[2], X[1] * Y[1]],
        1,
    )


def moment_matrices(K):
    """(6, 64) B_p with element matrix sum_p m_p B_p, exact for bilinear elements.

    Products of bilinear shape function derivatives are spanned by 1, x, y, x^2, y^2,
    xy, so a least squares fit on 8 x 8 voxels recovers the B_p exactly.
    """
    G = np.stack([T.T @ K @ T for T in child_transfers(8)]).reshape(64, 64)
    return np.linalg.lstsq(moment_weights(8), G, rcond=None)[0]


def child_transfers(children):
    """(children^2, 8, 8) bilinear maps from an element's dofs to each sub-element's."""
    T = np.zeros((children, children, 8, 8))
    for ka in range(children):
        for kb in range(children):
            for m in range(4):
                x = (ka + (m & 1)) / children
                y = (kb + (m >> 1)) / children
                for M in range(4):
                    w = (x if M & 1 else 1 - x) * (y if M >> 1 else 1 - y)
                    T[ka, kb, 2 * m, 2 * M] = w
                    T[ka, kb, 2 * m + 1, 2 * M + 1] = w
    return T.reshape(-1, 8, 8)


def group_children(field, children):
    """(ex c, ey c, ...) -> (ex ey, c^2 ...), children ordered as in `child_transfers`."""
    ex, ey = field.shape[0] // children, field.shape[1] // children
    rest = field.shape[2:]
    grouped = field.reshape(ex, children, ey, children, *rest).swapaxes(1, 2)
    return grouped.reshape(ex * ey, -1)


class Level:
    """one float32 grid of the hierarchy: voxel coefficients (CH > 0) or stored matrices."""

    def __init__(self, ex, ey, CH, mask):
        self.ex, self.ey, self.CH = ex, ey, CH
        self.nn = (ex + 1) * (ey + 1)
        self.ndof = 2 * self.nn
        self.mask = mask
        self.b, self.x, self.r, self.d, self.d2 = (
            cp.zeros(self.ndof, cp.float32) for _ in range(5)
        )
        self.dinv = cp.ones(self.ndof, cp.float32)
        self._power_vector = None

    def args(self, coeff, mats):
        self.coeff, self.mats = coeff, mats
        self.common = (
            np.int32(self.ex),
            np.int32(self.ey),
            np.int32(coeff.shape[1]),
            coeff,
            mats,
            self.mask,
        )

    def node_kernel(self, name, *args):
        launch(kernel(f"{name}<float, {self.CH}>"), self.nn, self.common + args)

    def apply(self, u, out):
        self.node_kernel("apply", u, out)
        return out

    def set_diagonal(self):
        self.node_kernel("diagonal", self.d2)
        cp.divide(1.0, self.d2, out=self.dinv)

    def estimate_lmax(self, iters):  # power iteration on D^-1 A, warm-started
        v, Av = self.d, self.d2
        if self._power_vector is None:
            rng = cp.random.default_rng(0)
            self._power_vector = rng.random(self.ndof, dtype=cp.float32)
            iters *= 3
        v[...] = self._power_vector
        _mask(self.mask, v)
        for _ in range(iters):
            v *= cp.dot(v, v) ** -0.5
            cp.multiply(self.dinv, self.apply(v, Av), out=v)
        self._power_vector[...] = v
        return float(cp.dot(v, v)) ** 0.5

    def dense(self):  # float64 dense matrix of a stored-matrix level
        ny = self.ey + 1
        ei, ej = np.meshgrid(np.arange(self.ex), np.arange(self.ey), indexing="ij")
        nodes = [(ei + (m & 1)) * ny + ej + (m >> 1) for m in range(4)]
        efts = np.stack([n.ravel() + f * self.nn for n in nodes for f in range(2)], 1)
        K_e = cp.asnumpy(self.mats).astype(np.float64).reshape(-1, 8, 8)
        dense = np.zeros((self.ndof, self.ndof))
        np.add.at(dense, (efts[:, :, None], efts[:, None, :]), K_e)
        m = cp.asnumpy(self.mask).astype(np.float64)
        dense = m[:, None] * dense * m[None, :]
        dense[np.diag_indices(self.ndof)] += 1.0 - m
        return dense


class StructuredMultigrid:
    """multigrid-preconditioned CG for plane stress on a voxel grid of bilinear elements.

    Nothing but the voxel coefficients is stored on the `matrix_free` finest levels:
    level d applies sum_k c_k T_k^T K T_k over the s 2^d x s 2^d voxels of each element,
    with s = `sub_voxels` voxels per fine element edge. Without matrix-free levels the
    fine level applies K_e = sum_p m_p B_p from six material moments per element. The
    next level stores its Galerkin element matrices, coarser levels from their
    children. Coarse Dirichlet masks are injected from the coincident fine node, which
    is exact when fixed dofs sit on coarse nodes or cover whole faces. CG keeps x and r
    in float64 and the rest in float32 with float64 products, and one iteration,
    V-cycle included, is replayed as a single CUDA graph.

    Args:
        nvoxels, lengths: voxel grid (divisible by s 2^matrix_free) and domain size.
        nu: Poisson ratio, Young's modulus is the voxel coefficient.
        fixed: bool array (2, nx / s + 1, ny / s + 1), True on Dirichlet dofs.
        smoothing: Chebyshev degree per level from fine to coarse (last repeats).
        coarse_dofs: stop coarsening once a level has at most this many dofs.
        matrix_free: number of fine levels applied from the voxel coefficients, by
            default those with elements of at most 4 x 4 voxels, and none (moments)
            from 3 x 3 voxel elements on.
        sub_voxels: voxels per fine element edge.
    """

    def __init__(
        self,
        nvoxels,
        lengths,
        nu,
        fixed,
        smoothing=(1, 2, 4, 8),
        coarse_dofs=300,
        eig_ratio=30.0,
        power_iters=8,
        matrix_free=None,
        sub_voxels=1,
    ):
        NX, NY = nvoxels
        sub = sub_voxels
        if matrix_free is None:
            matrix_free = sum(sub * 2**d <= 4 for d in range(3)) if sub < 3 else 0
        assert NX % (sub * 2**matrix_free) == 0 and NY % (sub * 2**matrix_free) == 0
        self.nvoxels, self.sub = (NX, NY), sub
        self.eig_ratio, self.power_iters = eig_ratio, power_iters
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.graph = None
        self.coarse_inverse = None
        K = element_stiffness([L / n for L, n in zip(lengths, nvoxels)], nu)
        self.coeff = cp.zeros((NX, NY), cp.float32)

        mask = cp.asarray(~np.asarray(fixed), dtype=cp.uint8)
        self.mask = mask.ravel()
        transfers = [child_transfers(sub * 2**d) for d in range(matrix_free + 1)]
        G = [np.stack([T.T @ K @ T for T in Ts]) for Ts in transfers]
        self.G64 = cp.asarray(G[0])
        self.levels = [
            Level(NX // sub, NY // sub, sub if matrix_free else -1, self.mask)
        ]
        fine = self.levels[0]
        if matrix_free:
            fine.args(self.coeff, cp.asarray(G[0], dtype=cp.float32))
            self.outer = (sub, self.coeff, self.G64)
        else:
            B = moment_matrices(K)
            self.moments = cp.zeros((fine.ex, fine.ey, 6), cp.float32)
            self.moment_weights = cp.asarray(moment_weights(sub), dtype=cp.float32)
            fine.args(self.moments, cp.asarray(B, dtype=cp.float32))
            self.outer = (-1, self.moments, cp.asarray(B))
            B = B.reshape(6, 8, 8)
            G_moments = [(T.T @ B @ T).reshape(6, 64) for T in child_transfers(2)]
            self.G_moments = cp.asarray(np.concatenate(G_moments), dtype=cp.float32)
        while True:
            fine = self.levels[-1]
            if fine.ex % 2 or fine.ey % 2 or fine.ndof <= coarse_dofs:
                break
            mask = mask[:, ::2, ::2]
            depth = len(self.levels)
            CH = sub * 2**depth if depth < matrix_free else 0
            level = Level(fine.ex // 2, fine.ey // 2, CH, mask.ravel())
            if CH:
                level.args(self.coeff, cp.asarray(G[depth], dtype=cp.float32))
            else:
                level.args(self.coeff, cp.zeros((level.ex, level.ey, 64), cp.float32))
            self.levels.append(level)
        assert len(self.levels) > matrix_free and self.levels[-1].CH == 0
        self.matrix_free = matrix_free
        self.G_first = cp.asarray(G[matrix_free].reshape(-1, 64), dtype=cp.float32)
        kron = [np.kron(T, T) for T in child_transfers(2)]
        self.G_children = cp.asarray(np.concatenate(kron), dtype=cp.float32)

        smoothing = [smoothing] if np.isscalar(smoothing) else list(smoothing)
        for i, lv in enumerate(self.levels):
            lv.smoothing = smoothing[min(i, len(smoothing) - 1)]
            lv.cheb = cp.zeros(2 * lv.smoothing - 1, cp.float32)

        # persistent CG state, z is the V-cycle output; scalars hold rz, pAp, rr,
        # rz_old, rz_new
        self.ndof = self.levels[0].ndof
        self.x, self.r = cp.zeros(self.ndof), cp.zeros(self.ndof)
        self.p, self.Ap = (
            cp.zeros(self.ndof, cp.float32),
            cp.zeros(self.ndof, cp.float32),
        )
        self.scalars = cp.zeros(5)
        self.partial = cp.zeros(DOT_BLOCKS)

    def _apply64(self, u, out):  # float64 fine operator of the outer CG
        U, V = ("float" if a.dtype == cp.float32 else "double" for a in (u, out))
        fine = self.levels[0]
        args = (np.int32(fine.ex), np.int32(fine.ey), np.int32(self.nvoxels[1]))
        CH, coeff, mats = self.outer
        args += (coeff, mats, self.mask, u, out)
        launch(kernel(f"apply64<{U}, {V}, {CH}>"), fine.nn, args)

    def update(self, coeff):
        """set up operator and hierarchy for a new voxel coefficient grid."""
        cp.cuda.get_current_stream().synchronize()
        with self.stream:
            self._update(coeff)
        self.stream.synchronize()

    def _update(self, coeff):
        self.coeff[...] = coeff
        levels = self.levels
        first = max(self.matrix_free, 1)
        if self.matrix_free:
            voxels = group_children(self.coeff, self.sub * 2**self.matrix_free)
            cp.matmul(voxels, self.G_first, out=levels[first].mats.reshape(-1, 64))
        else:
            voxels = group_children(self.coeff, self.sub)
            cp.matmul(voxels, self.moment_weights, out=self.moments.reshape(-1, 6))
            children = group_children(self.moments, 2)
            cp.matmul(children, self.G_moments, out=levels[1].mats.reshape(-1, 64))
        for fine, coarse in zip(levels[first:], levels[first + 1 :]):
            children = group_children(fine.mats, 2)
            cp.matmul(children, self.G_children, out=coarse.mats.reshape(-1, 64))
        for lv in levels[:-1]:
            lv.set_diagonal()
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
            lv.cheb[...] = cp.asarray(cheb, dtype=cp.float32)
        inverse = cp.linalg.inv(cp.asarray(levels[-1].dense()))
        if self.coarse_inverse is None:
            self.coarse_inverse = inverse
        else:
            self.coarse_inverse[...] = inverse

    def _smooth(self, lv, pre):
        """Chebyshev-Jacobi on lv.x for rhs lv.b; pre-smoothing starts from x = 0."""
        d, d_next = lv.d, lv.d2
        if pre:
            _pre_first(lv.b, lv.dinv, lv.cheb, lv.r, d, lv.x)
        else:
            lv.node_kernel("residual", lv.b, lv.x, lv.dinv, lv.cheb, lv.r, d)
            if lv.smoothing == 1:
                _add(d, lv.x)
        for k in range(lv.smoothing - 1):
            add_in = np.int32(not pre and k == 0)
            lv.node_kernel(
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
            lv.node_kernel("update_residual", d, lv.r)

    def _vcycle(self):  # levels[0].b -> levels[0].x
        L = len(self.levels) - 1
        for i in range(L):
            fine, coarse = self.levels[i], self.levels[i + 1]
            self._smooth(fine, pre=True)
            launch(
                kernel("restrict_<float>"),
                coarse.nn,
                (np.int32(fine.ex), np.int32(fine.ey), fine.r, coarse.b, coarse.mask),
            )
        coarse = self.levels[L]
        launch(
            kernel("dense_matvec<float>"),
            coarse.ndof,
            (self.coarse_inverse, coarse.b, coarse.x, np.int32(coarse.ndof)),
        )
        for i in reversed(range(L)):
            fine, coarse = self.levels[i], self.levels[i + 1]
            launch(
                kernel("prolong<float>"),
                fine.nn,
                (np.int32(fine.ex), np.int32(fine.ey), coarse.x, fine.x, fine.mask),
            )
            self._smooth(fine, pre=False)

    def _dot(self, a, b, out):  # deterministic two-stage reduction
        A, B = ("float" if v.dtype == cp.float32 else "double" for v in (a, b))
        partial = kernel(f"dot_partial<{A}, {B}>")
        partial((DOT_BLOCKS,), (256,), (a, b, self.partial, np.int64(a.size)))
        final = kernel("dot_final")
        final((1,), (1,), (self.partial, out, np.int32(DOT_BLOCKS)))

    def _precondition(self):  # r -> z = levels[0].x
        _cast(self.r, self.levels[0].b)
        self._vcycle()

    def _iteration(self):  # one flexible PCG step on the persistent state
        x, r, p, Ap, z = self.x, self.r, self.p, self.Ap, self.levels[0].x
        s = self.scalars
        self._apply64(p, Ap)  # float32 arithmetic here breaks CG at 1e-9 void stiffness
        self._dot(p, Ap, s[1])
        _xr_update(p, Ap, s, x, r)
        self._dot(r, r, s[2])
        self._dot(r, z, s[3])
        self._precondition()
        self._dot(r, z, s[4])
        _p_update(z, s, p)  # flexible beta, the float32 V-cycle is inexact
        s[0:1] = s[4:5]

    def _restart(self, rhs):  # true residual r = b - A x, p = z
        r, p, z = self.r, self.p, self.levels[0].x
        self._apply64(self.x, r)
        _init_residual(rhs, self.mask, r)
        self._precondition()
        p[...] = z
        self._dot(r, z, self.scalars[0])
        self._dot(r, r, self.scalars[2])

    def solve(self, rhs, x0=None, rtol=1e-8, maxiter=500, check=20):
        """flexible PCG on the fine operator; returns (u, iterations).

        u is the solver's float64 buffer, overwritten by the next solve. CG restarts
        from the true residual when its best norm drops less than 2x over `check`
        iterations: with 1e-9 void the coarse inverse amplifies float32 rounding into huge void
        displacements, pAp blows up and CG can stall for hundreds of iterations.
        """
        cp.cuda.get_current_stream().synchronize()
        x = self.x
        with self.stream:
            if x0 is None:
                x.fill(0.0)
            elif x0 is not x:
                x[...] = x0
            _mask(self.mask, x)
            self._restart(rhs)
            tol = rtol**2 * float(_masked_norm2(rhs, self.mask))
            if self.graph is None:
                self.stream.begin_capture()
                self._iteration()
                self.graph = self.stream.end_capture()
            best = best_check = float(self.scalars[2])
            for it in range(1, maxiter + 1):
                self.graph.launch(self.stream)
                rr = float(self.scalars[2])
                if rr <= tol:
                    break
                best = min(best, rr)
                if it % check == 0:
                    if best > 0.25 * best_check:
                        self._restart(rhs)
                    best_check = best
        self.stream.synchronize()
        return x, it

    def element_energy(self, u):
        """(NX, NY) float32 unit-coefficient energy per voxel."""
        fine = self.levels[0]
        out = cp.empty(self.nvoxels, cp.float32)
        G = self.G64.astype(cp.float32)
        args = (np.int32(fine.ex), np.int32(fine.ey), np.int32(self.sub), G)
        launch(kernel("energy"), out.size, args + (u, out))
        return out
