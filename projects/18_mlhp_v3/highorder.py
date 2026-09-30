"""multigrid-preconditioned CG for 2D voxel elasticity with condensed high-order elements."""

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage

# fine level: the element-edge skeleton of Q_p elements of s x s voxels after static
# condensation; its nodes are the element vertices, then the p - 1 nodes of every
# horizontal edge, then of every vertical edge, dofs (field, node). Element kernels
# write S_e u_e per element, node kernels sum the two or four element contributions
# of each node, so nothing is accumulated with atomics
SKELETON = r"""
template <int P>
__device__ __forceinline__ long long boundary_node(const int ex, const int ey, const int i,
                                                   const int j, const int l)
{
    // global skeleton node of local boundary node l of element (i, j); local order:
    // vertices (0,0), (p,0), (0,p), (p,p), then the bottom, top, left, right edges
    const long long V = (long long)(ex + 1) * (ey + 1);
    const long long H = (long long)ex * (ey + 1) * (P - 1);
    if (l < 4) return (long long)(i + (l & 1)) * (ey + 1) + j + (l >> 1);
    const int side = (l - 4) / (P - 1), k = (l - 4) % (P - 1);
    if (side == 0) return V + ((long long)i * (ey + 1) + j) * (P - 1) + k;
    if (side == 1) return V + ((long long)i * (ey + 1) + j + 1) * (P - 1) + k;
    if (side == 2) return V + H + ((long long)i * ey + j) * (P - 1) + k;
    return V + H + ((long long)(i + 1) * ey + j) * (P - 1) + k;
}

template <int P>
__device__ __forceinline__ long long skeleton_nodes(const int ex, const int ey)
{
    return (long long)(ex + 1) * (ey + 1) + (long long)ex * (ey + 1) * (P - 1)
           + (long long)(ex + 1) * ey * (P - 1);
}

// Y_e = S_e u_e from the packed upper triangle S[k E + e]. With PROJ the element rigid
// motions are removed from u_e in float64 before the float32 product and from the
// result after it: rounding S_e to float32 would otherwise stiffen rigid motion,
// which dominates the element displacements of a bending structure
template <int P, typename U, bool PROJ>
__global__ void element_apply(const int ex, const int ey, const float* S, const U* u,
                              float* Y, const double* xy, const double rot)
{
    constexpr int NB = 4 * P, ND = 2 * NB;
    const long long E = (long long)ex * ey;
    const long long e = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= E) return;
    const int i = e / ey, j = e % ey;
    const long long ns = skeleton_nodes<P>(ex, ey);
    long long g[NB];
    #pragma unroll
    for (int l = 0; l < NB; ++l) g[l] = boundary_node<P>(ex, ey, i, j, l);
    double tx = 0, ty = 0, w = 0;
    if (PROJ) {
        #pragma unroll
        for (int l = 0; l < NB; ++l) {
            const double ux = (double)u[g[l]], uy = (double)u[ns + g[l]];
            tx += ux;
            ty += uy;
            w += xy[l] * uy - xy[NB + l] * ux;
        }
        tx /= NB;
        ty /= NB;
        w *= rot;
    }
    float v[ND];
    #pragma unroll
    for (int l = 0; l < NB; ++l) {
        v[l] = (float)((double)u[g[l]] - (tx - w * xy[NB + l]));
        v[NB + l] = (float)((double)u[ns + g[l]] - (ty + w * xy[l]));
    }
    float y[ND];
    #pragma unroll
    for (int r = 0; r < ND; ++r) y[r] = 0;
    int k = 0;
    #pragma unroll
    for (int r = 0; r < ND; ++r) {
        #pragma unroll
        for (int c = r; c < ND; ++c, ++k) {
            const float s = S[(long long)k * E + e];
            y[r] += s * v[c];
            if (c != r) y[c] += s * v[r];
        }
    }
    if (PROJ) {
        float fx = 0, fy = 0, m = 0;
        #pragma unroll
        for (int l = 0; l < NB; ++l) {
            fx += y[l];
            fy += y[NB + l];
            m += (float)xy[l] * y[NB + l] - (float)xy[NB + l] * y[l];
        }
        fx /= NB;
        fy /= NB;
        m *= (float)rot;
        #pragma unroll
        for (int l = 0; l < NB; ++l) {
            y[l] -= fx - m * (float)xy[NB + l];
            y[NB + l] -= fy + m * (float)xy[l];
        }
    }
    #pragma unroll
    for (int r = 0; r < ND; ++r) Y[(long long)r * E + e] = y[r];
}

template <int P>
__global__ void element_diagonal(const int ex, const int ey, const float* S, float* Y)
{
    constexpr int ND = 8 * P;
    const long long E = (long long)ex * ey;
    const long long e = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= E) return;
    #pragma unroll
    for (int r = 0; r < ND; ++r) Y[(long long)r * E + e] = S[(long long)(r * ND - r * (r - 1) / 2) * E + e];
}

// sum of the element contributions Y of skeleton node t
template <int P, typename A>
__device__ __forceinline__ void gather_node(const int ex, const int ey, const float* Y,
                                            const long long t, A acc[2])
{
    constexpr int NB = 4 * P;
    const long long E = (long long)ex * ey;
    const long long V = (long long)(ex + 1) * (ey + 1);
    const long long H = (long long)ex * (ey + 1) * (P - 1);
    acc[0] = 0;
    acc[1] = 0;
    long long es[4];
    int ls[4], n = 0;
    if (t < V) {
        const int I = t / (ey + 1), J = t % (ey + 1);
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                const int ei = I - 1 + a, ej = J - 1 + b;
                if (ei < 0 || ej < 0 || ei >= ex || ej >= ey) continue;
                es[n] = (long long)ei * ey + ej;
                ls[n++] = (1 - a) + 2 * (1 - b);
            }
    } else if (t < V + H) {
        const long long q = t - V, edge = q / (P - 1);
        const int k = q % (P - 1), i = edge / (ey + 1), J = edge % (ey + 1);
        if (J > 0) { es[n] = (long long)i * ey + J - 1; ls[n++] = 4 + (P - 1) + k; }
        if (J < ey) { es[n] = (long long)i * ey + J; ls[n++] = 4 + k; }
    } else {
        const long long q = t - V - H, edge = q / (P - 1);
        const int k = q % (P - 1), I = edge / ey, j = edge % ey;
        if (I > 0) { es[n] = (long long)(I - 1) * ey + j; ls[n++] = 4 + 3 * (P - 1) + k; }
        if (I < ex) { es[n] = (long long)I * ey + j; ls[n++] = 4 + 2 * (P - 1) + k; }
    }
    for (int q = 0; q < n; ++q) {
        acc[0] += (A)Y[(long long)ls[q] * E + es[q]];
        acc[1] += (A)Y[(long long)(NB + ls[q]) * E + es[q]];
    }
}

// OP 0: out = A a; 1: out -= A a; 2: out = b - A a; 3: out = diag A. Fixed dofs act
// as identity rows
template <int P, typename A, typename V, int OP>
__global__ void gather(const int ex, const int ey, const float* Y, const unsigned char* mask,
                       const V* a, const V* b, V* out)
{
    const long long ns = skeleton_nodes<P>(ex, ey);
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns) return;
    A acc[2];
    gather_node<P, A>(ex, ey, Y, t, acc);
    for (int f = 0; f < 2; ++f) {
        const long long k = f * ns + t;
        const A au = mask[k] ? acc[f] : (OP == 3 ? (A)1 : (A)a[k]);
        if (OP == 0 || OP == 3) out[k] = (V)au;
        if (OP == 1) out[k] = (V)((A)out[k] - au);
        if (OP == 2) out[k] = (V)((A)b[k] - au);
    }
}

// xf += mask P xc: vertices are copied, edge nodes interpolated linearly between the
// vertices of their edge at the Gauss-Lobatto positions tk
template <int P, typename T>
__global__ void skeleton_prolong(const int ex, const int ey, const T* xc, T* xf,
                                 const unsigned char* mask, const float* tk)
{
    const long long ns = skeleton_nodes<P>(ex, ey);
    const long long V = (long long)(ex + 1) * (ey + 1);
    const long long H = (long long)ex * (ey + 1) * (P - 1);
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns) return;
    long long v0 = t, v1 = t;
    float w = 0;
    if (t >= V && t < V + H) {
        const long long q = t - V, edge = q / (P - 1);
        const int i = edge / (ey + 1), J = edge % (ey + 1);
        w = tk[q % (P - 1)];
        v0 = (long long)i * (ey + 1) + J;
        v1 = v0 + ey + 1;
    } else if (t >= V + H) {
        const long long q = t - V - H, edge = q / (P - 1);
        const int I = edge / ey, j = edge % ey;
        w = tk[q % (P - 1)];
        v0 = (long long)I * (ey + 1) + j;
        v1 = v0 + 1;
    }
    for (int f = 0; f < 2; ++f) {
        const T v = (1 - w) * xc[f * V + v0] + (t < V ? (T)0 : w * xc[f * V + v1]);
        const long long k = f * ns + t;
        xf[k] += mask[k] ? v : (T)0;
    }
}

// bc = mask (P^T rf) on the vertex grid
template <int P, typename T>
__global__ void skeleton_restrict(const int ex, const int ey, const T* rf, T* bc,
                                  const unsigned char* mask, const float* tk)
{
    const long long ns = skeleton_nodes<P>(ex, ey);
    const long long V = (long long)(ex + 1) * (ey + 1);
    const long long H = (long long)ex * (ey + 1) * (P - 1);
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= V) return;
    const int I = t / (ey + 1), J = t % (ey + 1);
    for (int f = 0; f < 2; ++f) {
        const T* r = rf + f * ns;
        T s = r[t];
        for (int k = 0; k < P - 1; ++k) {
            if (I > 0) s += tk[k] * r[V + ((long long)(I - 1) * (ey + 1) + J) * (P - 1) + k];
            if (I < ex) s += (1 - tk[k]) * r[V + ((long long)I * (ey + 1) + J) * (P - 1) + k];
            if (J > 0) s += tk[k] * r[V + H + ((long long)I * ey + J - 1) * (P - 1) + k];
            if (J < ey) s += (1 - tk[k]) * r[V + H + ((long long)I * ey + J) * (P - 1) + k];
        }
        bc[f * V + t] = mask[f * V + t] ? s : (T)0;
    }
}

// local boundary node of neighbour (side) that coincides with local node l, or -1;
// sides 0-3: bottom, top, left, right neighbour, 4-7: diagonal neighbours of vertex 0-3
template <int P>
__device__ __forceinline__ int neighbour_node(const int side, const int l)
{
    if (side >= 4) return l == side - 4 ? 3 - l : -1;
    if (l < 4) {
        const int x = l & 1, y = l >> 1;
        if (side == 0) return y == 0 ? l + 2 : -1;
        if (side == 1) return y == 1 ? l - 2 : -1;
        if (side == 2) return x == 0 ? l + 1 : -1;
        return x == 1 ? l - 1 : -1;
    }
    const int s = (l - 4) / (P - 1), k = (l - 4) % (P - 1);
    const int partner[4] = {1, 0, 3, 2};
    return s == side ? 4 + partner[s] * (P - 1) + k : -1;
}

// entry (r, c) of the packed symmetric ND x ND matrix M of element e
template <int ND>
__device__ __forceinline__ float packed(const float* M, const long long E, const long long e,
                                        int r, int c)
{
    if (r > c) { const int s = r; r = c; c = s; }
    return M[(long long)(r * ND - r * (r - 1) / 2 + c - r) * E + e];
}

// additive Schwarz: invert the element block of the assembled operator (own condensed
// matrix plus the neighbours sharing its edges and vertices) by Cholesky in float32
// after diagonal scaling; fixed dofs are identity rows
template <int P>
__global__ void schwarz_inverse(const int ex, const int ey, const float* S,
                                const unsigned char* mask, const int n,
                                const long long* index, float* B)
{
    constexpr int NB = 4 * P, ND = 2 * NB;
    const long long E = (long long)ex * ey;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n) return;
    const long long e = index[t];
    const int i = e / ey, j = e % ey;
    const long long ns = skeleton_nodes<P>(ex, ey);
    const int di[8] = {0, 0, -1, 1, -1, 1, -1, 1}, dj[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    bool free[ND];
    for (int r = 0; r < ND; ++r)
        free[r] = mask[(r / NB) * ns + boundary_node<P>(ex, ey, i, j, r % NB)];
    float A[ND * ND];
    for (int r = 0; r < ND; ++r) {
        for (int c = r; c < ND; ++c) {
            float a = packed<ND>(S, E, e, r, c);
            for (int side = 0; side < 8; ++side) {
                const int ni = i + di[side], nj = j + dj[side];
                if (ni < 0 || nj < 0 || ni >= ex || nj >= ey) continue;
                const int mr = neighbour_node<P>(side, r % NB);
                const int mc = neighbour_node<P>(side, c % NB);
                if (mr < 0 || mc < 0) continue;
                a += packed<ND>(S, E, (long long)ni * ey + nj, (r / NB) * NB + mr,
                                (c / NB) * NB + mc);
            }
            if (!free[r] || !free[c]) a = r == c ? 1.0f : 0.0f;
            A[r * ND + c] = A[c * ND + r] = a;
        }
    }
    float scale[ND];
    for (int r = 0; r < ND; ++r) scale[r] = rsqrtf(A[r * ND + r]);
    for (int r = 0; r < ND; ++r)
        for (int c = 0; c < ND; ++c) A[r * ND + c] *= scale[r] * scale[c];
    // Cholesky A = L L^T in the lower triangle, pivots guarded against void modes
    for (int c = 0; c < ND; ++c) {
        float d = A[c * ND + c];
        for (int k = 0; k < c; ++k) d -= A[c * ND + k] * A[c * ND + k];
        d = sqrtf(fmaxf(d, 1e-10f));
        A[c * ND + c] = d;
        for (int r = c + 1; r < ND; ++r) {
            float a = A[r * ND + c];
            for (int k = 0; k < c; ++k) a -= A[r * ND + k] * A[c * ND + k];
            A[r * ND + c] = a / d;
        }
    }
    // L^-1 in place, columns from the right
    float tmp[ND];
    for (int c = ND - 1; c >= 0; --c) {
        const float inv = 1.0f / A[c * ND + c];
        for (int r = c + 1; r < ND; ++r) {
            float a = 0;
            for (int k = c + 1; k <= r; ++k) a += A[r * ND + k] * A[k * ND + c];
            tmp[r] = -a * inv;
        }
        A[c * ND + c] = inv;
        for (int r = c + 1; r < ND; ++r) A[r * ND + c] = tmp[r];
    }
    // A^-1 = L^-T L^-1, unscaled and packed
    int k = 0;
    for (int r = 0; r < ND; ++r) {
        for (int c = r; c < ND; ++c, ++k) {
            float a = 0;
            for (int q = c; q < ND; ++q) a += A[q * ND + r] * A[q * ND + c];
            B[(long long)k * E + e] = a * scale[r] * scale[c];
        }
    }
}

// blocks of elements in a uniform neighbourhood are the unit block scaled
template <int P>
__global__ void scale_block(const long long E, const float* c, const bool* uniform,
                            const float* B_ref, float* B)
{
    constexpr int ND = 8 * P, NSYM = ND * (ND + 1) / 2;
    const long long e = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= E || !uniform[e]) return;
    const float inv = 1.0f / c[e];
    for (int k = 0; k < NSYM; ++k) B[(long long)k * E + e] = B_ref[k] * inv;
}

// static condensation of one chunk of elements: Cholesky of the interior block in
// float64, X = K_ii^-1 K_ib, packed S = K_bb - K_bi X, and the Galerkin matrix Q^T S Q
// of the bilinear level; K blocks come row-major per element from the float32 GEMMs
template <int P>
__global__ void condense(const long long E, const int n, const long long* index,
                         const float* Kbb, const float* Kbi, const float* Kii,
                         const double* Q, float* S, float* X, float* mats)
{
    constexpr int ND = 8 * P, NI = 2 * (P - 1) * (P - 1);
    const long long el = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (el >= n) return;
    const long long e = index[el];
    const float* kbb = Kbb + el * ND * ND;
    const float* kbi = Kbi + el * ND * NI;
    const float* kii = Kii + el * NI * NI;
    double L[NI * NI];
    for (int q = 0; q < NI * NI; ++q) L[q] = kii[q];
    for (int c = 0; c < NI; ++c) {
        double d = L[c * NI + c];
        for (int k = 0; k < c; ++k) d -= L[c * NI + k] * L[c * NI + k];
        d = sqrt(d);
        L[c * NI + c] = d;
        for (int r = c + 1; r < NI; ++r) {
            double a = L[r * NI + c];
            for (int k = 0; k < c; ++k) a -= L[r * NI + k] * L[c * NI + k];
            L[r * NI + c] = a / d;
        }
    }
    double Xm[NI * ND];
    for (int c = 0; c < ND; ++c) {
        double y[NI];
        for (int q = 0; q < NI; ++q) {
            double a = kbi[c * NI + q];
            for (int k = 0; k < q; ++k) a -= L[q * NI + k] * y[k];
            y[q] = a / L[q * NI + q];
        }
        for (int q = NI - 1; q >= 0; --q) {
            double a = y[q];
            for (int k = q + 1; k < NI; ++k) a -= L[k * NI + q] * Xm[k * ND + c];
            Xm[q * ND + c] = a / L[q * NI + q];
        }
    }
    for (int q = 0; q < NI * ND; ++q) X[(long long)q * E + e] = (float)Xm[q];
    double T[ND * 8];
    for (int q = 0; q < ND * 8; ++q) T[q] = 0;
    int k = 0;
    for (int r = 0; r < ND; ++r) {
        for (int c = r; c < ND; ++c, ++k) {
            double a = kbb[r * ND + c];
            for (int q = 0; q < NI; ++q) a -= (double)kbi[r * NI + q] * Xm[q * ND + c];
            S[(long long)k * E + e] = (float)a;
            for (int b = 0; b < 8; ++b) {
                T[r * 8 + b] += a * Q[c * 8 + b];
                if (c != r) T[c * 8 + b] += a * Q[r * 8 + b];
            }
        }
    }
    for (int a = 0; a < 8; ++a) {
        for (int b = 0; b < 8; ++b) {
            double m = 0;
            for (int r = 0; r < ND; ++r) m += Q[r * 8 + a] * T[r * 8 + b];
            mats[e * 64 + a * 8 + b] = (float)m;
        }
    }
}

// elements of uniform voxels (all solid or all void) are the unit element scaled
template <int P>
__global__ void scale_reference(const long long E, const float* cmin, const float* cmax,
                                const float* S_ref, const float* X_ref,
                                const float* mats_ref, float* S, float* X, float* mats)
{
    constexpr int ND = 8 * P, NI = 2 * (P - 1) * (P - 1), NSYM = ND * (ND + 1) / 2;
    const long long e = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= E || cmin[e] != cmax[e]) return;
    const float c = cmin[e];
    for (int k = 0; k < NSYM; ++k) S[(long long)k * E + e] = c * S_ref[k];
    for (int q = 0; q < NI * ND; ++q) X[(long long)q * E + e] = X_ref[q];
    for (int q = 0; q < 64; ++q) mats[e * 64 + q] = c * mats_ref[q];
}

// element deformation W (rigid motion removed in float64), interior nodes from X
template <int P>
__global__ void deformation(const int ex, const int ey, const double* u, const float* X,
                            float* W, const double* xy, const double rot, const int* full)
{
    constexpr int NB = 4 * P, ND = 2 * NB, NI = 2 * (P - 1) * (P - 1);
    const long long E = (long long)ex * ey;
    const long long e = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= E) return;
    const int i = e / ey, j = e % ey;
    const long long ns = skeleton_nodes<P>(ex, ey);
    double tx = 0, ty = 0, w = 0;
    for (int l = 0; l < NB; ++l) {
        const long long g = boundary_node<P>(ex, ey, i, j, l);
        tx += u[g];
        ty += u[ns + g];
        w += xy[l] * u[ns + g] - xy[NB + l] * u[g];
    }
    tx /= NB;
    ty /= NB;
    w *= rot;
    float v[ND];
    for (int l = 0; l < NB; ++l) {
        const long long g = boundary_node<P>(ex, ey, i, j, l);
        v[l] = (float)(u[g] - (tx - w * xy[NB + l]));
        v[NB + l] = (float)(u[ns + g] - (ty + w * xy[l]));
        W[(long long)full[l] * E + e] = v[l];
        W[(long long)full[NB + l] * E + e] = v[NB + l];
    }
    for (int q = 0; q < NI; ++q) {
        float s = 0;
        for (int c = 0; c < ND; ++c) s -= X[(long long)(q * ND + c) * E + e] * v[c];
        W[(long long)full[ND + q] * E + e] = s;
    }
}

// energy of each voxel from the element deformation W, integrated with (p + 1)^2
// Gauss points by sum factorization; V1, D1[k][q][a] are the 1D basis values and
// physical derivatives at Gauss point q of voxel row k, wq the scaled Gauss weights
template <int P>
__global__ void voxel_energy(const int ex, const int ey, const int sub, const float* V1,
                             const float* D1, const float* wq, const float d11,
                             const float d12, const float d33, const float* W, float* out)
{
    constexpr int M = P + 1;
    const long long E = (long long)ex * ey;
    const int vy = ey * sub;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= (long long)ex * sub * vy) return;
    const int vi = t / vy, vj = t % vy;
    const long long e = (long long)(vi / sub) * ey + vj / sub;
    const float *Vx = V1 + (vi % sub) * M * M, *Dx = D1 + (vi % sub) * M * M;
    const float *Vy = V1 + (vj % sub) * M * M, *Dy = D1 + (vj % sub) * M * M;
    float g[2][2][M][M];  // g[field][derivative][qx][qy]
    for (int f = 0; f < 2; ++f) {
        float Tv[M][M], Td[M][M];  // [a][qy]
        for (int a = 0; a < M; ++a) {
            float w[M];
            for (int b = 0; b < M; ++b) w[b] = W[(long long)(2 * (a * M + b) + f) * E + e];
            for (int q = 0; q < M; ++q) {
                float sv = 0, sd = 0;
                for (int b = 0; b < M; ++b) {
                    sv += Vy[q * M + b] * w[b];
                    sd += Dy[q * M + b] * w[b];
                }
                Tv[a][q] = sv;
                Td[a][q] = sd;
            }
        }
        for (int qx = 0; qx < M; ++qx) {
            for (int qy = 0; qy < M; ++qy) {
                float gx = 0, gy = 0;
                for (int a = 0; a < M; ++a) {
                    gx += Dx[qx * M + a] * Tv[a][qy];
                    gy += Vx[qx * M + a] * Td[a][qy];
                }
                g[f][0][qx][qy] = gx;
                g[f][1][qx][qy] = gy;
            }
        }
    }
    float s = 0;
    for (int qx = 0; qx < M; ++qx) {
        for (int qy = 0; qy < M; ++qy) {
            const float exx = g[0][0][qx][qy], eyy = g[1][1][qx][qy];
            const float gxy = g[0][1][qx][qy] + g[1][0][qx][qy];
            s += wq[qx] * wq[qy]
                 * (d11 * (exx * exx + eyy * eyy) + 2 * d12 * exx * eyy + d33 * gxy * gxy);
        }
    }
    out[t] = s;
}
"""

# coarse levels: bilinear elements with stored element matrices, as in 18_mlhp_v2
BILINEAR = r"""
#define LEVEL_ARGS const int ex, const int ey, const float* mats, const unsigned char* mask
#define LEVEL ex, ey, mats, mask

template <typename T, bool DIAG>
__device__ __forceinline__ void node_rows(LEVEL_ARGS, const T* u, const int i, const int j,
                                          T acc[2])
{
    const int ny = ey + 1;
    const long long nn = (long long)(ex + 1) * ny;
    acc[0] = 0;
    acc[1] = 0;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            const int ei = i - 1 + a, ej = j - 1 + b;
            if (ei < 0 || ej < 0 || ei >= ex || ej >= ey) continue;
            T ue[8];
            for (int m = 0; m < 4; ++m) {
                const long long n = (long long)(ei + (m & 1)) * ny + ej + (m >> 1);
                ue[2 * m] = DIAG ? (T)0 : u[n];
                ue[2 * m + 1] = DIAG ? (T)0 : u[nn + n];
            }
            const int l = (1 - a) + 2 * (1 - b);
            for (int f = 0; f < 2; ++f) {
                const int r = 2 * l + f;
                const float* Kr = mats + ((long long)ei * ey + ej) * 64 + r * 8;
                if (DIAG) {
                    acc[f] += Kr[r];
                } else {
                    for (int q = 0; q < 8; ++q) acc[f] += Kr[q] * ue[q];
                }
            }
        }
    }
}

#define NODE_KERNEL(name, args, body)                                              \
template <typename T>                                                              \
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
#define ROWS(u) node_rows<T, false>(LEVEL, u, i, j, acc)
#define FOR_FIELDS(stmt) for (int f = 0; f < 2; ++f) { const long long k = f * nn + t; stmt }
#define AU(u) (mask[k] ? acc[f] : u[k])
#define COMMA ,

NODE_KERNEL(apply, const T* u COMMA T* out, ROWS(u); FOR_FIELDS(out[k] = AU(u);))

NODE_KERNEL(diagonal, T* out,
            node_rows<T COMMA true>(LEVEL, (const T*)0, i, j, acc);
            FOR_FIELDS(out[k] = mask[k] ? acc[f] : (T)1;))

NODE_KERNEL(residual,
            const T* b COMMA const T* x COMMA const T* dinv COMMA const T* cheb
            COMMA T* r COMMA T* d,
            ROWS(x); FOR_FIELDS(r[k] = b[k] - AU(x); d[k] = dinv[k] * r[k] * cheb[0];))

NODE_KERNEL(update_residual, const T* d COMMA T* r, ROWS(d); FOR_FIELDS(r[k] -= AU(d);))

NODE_KERNEL(chebyshev,
            const T* dinv COMMA const T* cheb COMMA const T* d_in COMMA T* d_out
            COMMA T* r COMMA T* x COMMA const int add_in,
            ROWS(d_in);
            FOR_FIELDS(const T ri = r[k] - AU(d_in); r[k] = ri;
                       const T dn = cheb[0] * d_in[k] + cheb[1] * dinv[k] * ri;
                       d_out[k] = dn; x[k] += add_in ? dn + d_in[k] : dn;))

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
    """compiled kernel or template instance, e.g. `kernel("gather<3, float, float, 0>")`."""
    if name not in _functions:
        module = cp.RawModule(code=SKELETON + BILINEAR, name_expressions=[name])
        _functions[name] = module.get_function(name)
    return _functions[name]


def launch(function, threads, args):
    function(((threads + 255) // 256,), (256,), args)


_add = cp.ElementwiseKernel("T a", "T x", "x += a", "add")
_cast = cp.ElementwiseKernel("S a", "T b", "b = (T)a", "cast")
_mask = cp.ElementwiseKernel("uint8 m", "T x", "x = m ? x : (T)0", "mask")
_pre_first = cp.ElementwiseKernel(
    "T b, T dinv, raw T cheb",
    "T r, T d, T x",
    "r = b; d = dinv * b * cheb[0]; x = d",
    "pre_first",
)
_cheb_update = cp.ElementwiseKernel(
    "T z, raw T cheb, int32 add_in",
    "T d, T x",
    "const T dn = cheb[0] * d + cheb[1] * z; x += add_in ? dn + d : dn; d = dn",
    "cheb_update",
)
_scaled = cp.ElementwiseKernel(
    "T z, raw T cheb", "T d, T x", "d = cheb[0] * z; x = d", "scaled"
)
_scale = cp.ElementwiseKernel("T z, raw T cheb", "T d", "d = cheb[0] * z", "scale")
_masked_norm2 = cp.ReductionKernel(
    "S b, uint8 m",
    "float64 y",
    "m ? (double)b * b : 0.0",
    "a + b",
    "y = a",
    "0",
    "masked_norm2",
)
_xr_update = cp.ElementwiseKernel(
    "F p, F Ap, raw D s",
    "D x, D r",
    "D alpha = s[0] / s[1]; x += alpha * p; r -= alpha * Ap",
    "xr_update",
)
_p_update = cp.ElementwiseKernel(
    "F z, raw D s", "F p", "p = z + (s[4] - s[3]) / s[0] * p", "p_update"
)


def gauss_lobatto(p):
    """p + 1 Gauss-Lobatto points on [-1, 1]."""
    c = np.zeros(p + 1)
    c[p] = 1.0
    inner = np.polynomial.legendre.legroots(np.polynomial.legendre.legder(c))
    return np.concatenate([[-1.0], np.sort(inner), [1.0]])


def lagrange(nodes, x):
    """values and derivatives of the Lagrange polynomials on `nodes` at points x."""
    V = np.ones((len(x), len(nodes)))
    D = np.zeros((len(x), len(nodes)))
    for a, xa in enumerate(nodes):
        others = np.delete(nodes, a)
        denom = np.prod(xa - others)
        V[:, a] = np.prod(x[:, None] - others, axis=1) / denom
        for c in range(len(others)):
            D[:, a] += np.prod(x[:, None] - np.delete(others, c), axis=1) / denom
    return V, D


def voxel_matrices(p, sub, h, nu):
    """(sub^2, Ne, Ne) unit-modulus plane stress stiffness of each voxel of a Q_p element.

    Element nodes (a, b) sit at the Gauss-Lobatto points, node a (p + 1) + b with a
    along x, dofs 2 node + field; voxel k = kx sub + ky. Each voxel is integrated
    exactly with (p + 1)^2 Gauss points.
    """
    nodes = gauss_lobatto(p)
    g, w = np.polynomial.legendre.leggauss(p + 1)
    D = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2]])
    D /= 1.0 - nu**2
    m = p + 1
    out = []
    for kx in range(sub):
        for ky in range(sub):
            Vx, Dx = lagrange(nodes, -1 + (2 * kx + g + 1) / sub)
            Vy, Dy = lagrange(nodes, -1 + (2 * ky + g + 1) / sub)
            K = np.zeros((2 * m * m, 2 * m * m))
            for qi in range(m):
                for qj in range(m):
                    Nx = np.outer(Dx[qi], Vy[qj]).ravel() * 2 / (sub * h)
                    Ny = np.outer(Vx[qi], Dy[qj]).ravel() * 2 / (sub * h)
                    B = np.zeros((3, 2 * m * m))
                    B[0, 0::2], B[1, 1::2] = Nx, Ny
                    B[2, 0::2], B[2, 1::2] = Ny, Nx
                    K += B.T @ D @ B * w[qi] * w[qj] * (h / 2) ** 2
            out.append(K)
    return np.array(out)


def boundary_layout(p):
    """local (a, b) of the 4p boundary nodes in kernel order and of the interior nodes."""
    edge = range(1, p)
    boundary = [(0, 0), (p, 0), (0, p), (p, p)]
    boundary += [(k, 0) for k in edge] + [(k, p) for k in edge]
    boundary += [(0, k) for k in edge] + [(p, k) for k in edge]
    interior = [(a, b) for a in edge for b in edge]
    return boundary, interior


def skeleton_nodes(ex, ey, p):
    """(i, j) positions on the (p ex + 1) x (p ey + 1) node grid of all skeleton nodes."""

    def grid(*n):
        return [a.ravel() for a in np.meshgrid(*map(np.arange, n), indexing="ij")]

    vi, vj = grid(ex + 1, ey + 1)
    hi, hj, hk = grid(ex, ey + 1, p - 1)
    wi, wj, wk = grid(ex + 1, ey, p - 1)
    vertices = (p * vi, p * vj)
    horizontal = (p * hi + hk + 1, p * hj)
    vertical = (p * wi, p * wj + wk + 1)
    return tuple(np.concatenate(c) for c in zip(vertices, horizontal, vertical))


def group_children(field, children):
    """(ex c, ey c, ...) -> (ex ey, c^2 ...), children ordered kx c + ky."""
    ex, ey = field.shape[0] // children, field.shape[1] // children
    rest = field.shape[2:]
    grouped = field.reshape(ex, children, ey, children, *rest).swapaxes(1, 2)
    return grouped.reshape(ex * ey, -1)


def bilinear_children():
    """(4, 8, 8) bilinear maps from an element's dofs to each of its 2 x 2 children."""
    T = np.zeros((2, 2, 8, 8))
    for ka in range(2):
        for kb in range(2):
            for m in range(4):
                x, y = (ka + (m & 1)) / 2, (kb + (m >> 1)) / 2
                for M in range(4):
                    w = (x if M & 1 else 1 - x) * (y if M >> 1 else 1 - y)
                    T[ka, kb, 2 * m, 2 * M] = T[ka, kb, 2 * m + 1, 2 * M + 1] = w
    return T.reshape(-1, 8, 8)


def power_iteration(level, iters):  # lambda_max of M^-1 A, warm-started
    v, Av = level.d, level.d2
    if level._power_vector is None:
        rng = cp.random.default_rng(0)
        level._power_vector = rng.random(level.ndof, dtype=cp.float32)
        iters *= 3
    v[...] = level._power_vector
    _mask(level.mask, v)
    for _ in range(iters):
        v *= cp.dot(v, v) ** -0.5
        level.apply(v, Av)
        level.precondition(Av, v)
    level._power_vector[...] = v
    return float(cp.dot(v, v)) ** 0.5


def chebyshev_coefficients(hi, eig_ratio, degree):
    lo = hi / eig_ratio
    theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
    sigma = theta / delta
    rho = 1.0 / sigma
    cheb = [1.0 / theta]
    for _ in range(degree - 1):
        rho_new = 1.0 / (2.0 * sigma - rho)
        cheb += [rho_new * rho, 2.0 * rho_new / delta]
        rho = rho_new
    return cheb


class BilinearLevel:
    """coarse float32 grid of bilinear elements with stored element matrices."""

    def __init__(self, ex, ey, mask):
        self.ex, self.ey = ex, ey
        self.nn = (ex + 1) * (ey + 1)
        self.ndof = 2 * self.nn
        self.mask = mask
        self.b, self.x, self.r, self.d, self.d2 = (
            cp.zeros(self.ndof, cp.float32) for _ in range(5)
        )
        self.dinv = cp.ones(self.ndof, cp.float32)
        self.mats = cp.zeros((ex * ey, 64), cp.float32)
        self.common = (np.int32(ex), np.int32(ey), self.mats, mask)
        self._power_vector = None

    def node_kernel(self, name, *args):
        launch(kernel(f"{name}<float>"), self.nn, self.common + args)

    def apply(self, u, out):
        self.node_kernel("apply", u, out)
        return out

    def precondition(self, r, z):
        cp.multiply(self.dinv, r, out=z)

    def set_diagonal(self):
        self.node_kernel("diagonal", self.d2)
        cp.divide(1.0, self.d2, out=self.dinv)

    def smooth(self, pre):
        """Chebyshev-Jacobi on x for rhs b; pre-smoothing starts from x = 0."""
        d, d_next = self.d, self.d2
        if pre:
            _pre_first(self.b, self.dinv, self.cheb, self.r, d, self.x)
        else:
            self.node_kernel(
                "residual", self.b, self.x, self.dinv, self.cheb, self.r, d
            )
            if self.smoothing == 1:
                _add(d, self.x)
        for k in range(self.smoothing - 1):
            add_in = np.int32(not pre and k == 0)
            self.node_kernel(
                "chebyshev",
                self.dinv,
                self.cheb[2 * k + 1 :],
                d,
                d_next,
                self.r,
                self.x,
                add_in,
            )
            d, d_next = d_next, d
        if pre:
            self.node_kernel("update_residual", d, self.r)

    def dense(self):  # float64 dense masked matrix
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


class SkeletonLevel:
    """fine float32 level: condensed Q_p elements on the element-edge skeleton."""

    def __init__(self, ex, ey, p, mask, schwarz, geometry):
        self.ex, self.ey, self.p = ex, ey, p
        self.E = ex * ey
        self.nn = len(skeleton_nodes(ex, ey, p)[0])
        self.ndof = 2 * self.nn
        self.mask = mask
        self.schwarz = schwarz
        self.b, self.x, self.r, self.d, self.d2, self.z = (
            cp.zeros(self.ndof, cp.float32) for _ in range(6)
        )
        self.dinv = cp.ones(self.ndof, cp.float32)
        nd = 8 * p
        self.S = cp.zeros((nd * (nd + 1) // 2, self.E), cp.float32)
        self.B = cp.zeros_like(self.S) if schwarz else None
        self.Y = cp.zeros((nd, self.E), cp.float32)
        self.xy, self.rot = geometry
        self._power_vector = None

    def _element(self, S, u, project):
        U = "double" if u.dtype == cp.float64 else "float"
        name = f"element_apply<{self.p}, {U}, {'true' if project else 'false'}>"
        args = (np.int32(self.ex), np.int32(self.ey), S, u, self.Y, self.xy, self.rot)
        launch(kernel(name), self.E, args)

    def _gather(self, op, a, b, out, acc="float"):
        V = "double" if out.dtype == cp.float64 else "float"
        args = (np.int32(self.ex), np.int32(self.ey), self.Y, self.mask, a, b, out)
        launch(kernel(f"gather<{self.p}, {acc}, {V}, {op}>"), self.nn, args)

    def apply(self, u, out, acc="float"):
        self._element(self.S, u, True)
        self._gather(0, u, u, out, acc)
        return out

    def residual(self, b, x, r, acc="float"):  # r = b - A x
        self._element(self.S, x, True)
        self._gather(2, x, b, r, acc)

    def update_residual(self, d, r):  # r -= A d
        self._element(self.S, d, True)
        self._gather(1, d, d, r)

    def precondition(self, r, z):
        if self.schwarz:
            self._element(self.B, r, False)
            self._gather(0, r, r, z)
        else:
            cp.multiply(self.dinv, r, out=z)

    def set_diagonal(self):
        launch(
            kernel(f"element_diagonal<{self.p}>"),
            self.E,
            (np.int32(self.ex), np.int32(self.ey), self.S, self.Y),
        )
        self._gather(3, self.d2, self.d2, self.d2)
        cp.divide(1.0, self.d2, out=self.dinv)

    def set_schwarz(self, cmin, cmax):
        """inverted element blocks of the assembled operator.

        Elements whose 3 x 3 neighbourhood is uniform and free of supports get the unit
        block `B_ref` scaled, the others are inverted by `schwarz_inverse`.
        """
        shape = (self.ex, self.ey)
        lo = ndimage.minimum_filter(cmin.reshape(shape), 3, mode="nearest")
        hi = ndimage.maximum_filter(cmax.reshape(shape), 3, mode="nearest")
        uniform = (lo == hi).ravel() & ~self.clamped
        args = (np.int64(self.E), cmin, uniform, self.B_ref, self.B)
        launch(kernel(f"scale_block<{self.p}>"), self.E, args)
        self.invert_blocks(cp.flatnonzero(~uniform))

    def invert_blocks(self, index):
        args = (np.int32(self.ex), np.int32(self.ey), self.S, self.mask)
        args += (np.int32(len(index)), index, self.B)
        launch(kernel(f"schwarz_inverse<{self.p}>"), len(index), args)

    def smooth(self, pre):
        """Chebyshev with Jacobi or additive Schwarz on x for rhs b."""
        r, d, z = self.r, self.d, self.z
        if pre:
            r[...] = self.b
            self.precondition(r, z)
            _scaled(z, self.cheb, d, self.x)
        else:
            self.residual(self.b, self.x, r)
            self.precondition(r, z)
            _scale(z, self.cheb, d)
            if self.smoothing == 1:
                _add(d, self.x)
        for k in range(self.smoothing - 1):
            self.update_residual(d, r)
            self.precondition(r, z)
            _cheb_update(
                z, self.cheb[2 * k + 1 :], np.int32(not pre and k == 0), d, self.x
            )
        if pre:
            self.update_residual(d, r)


class HighOrderMultigrid:
    """multigrid-preconditioned CG for plane stress with condensed Q_p voxel elements.

    Each element holds s x s voxels and a full tensor Q_p space on Gauss-Lobatto nodes.
    Its interior nodes are condensed out once per design update, so CG runs on the
    element-edge skeleton, about 2 (2 p - 1) / s^2 dofs per voxel. The condensed
    element matrices are stored in float32 and applied with the element rigid motions
    removed in float64, which rounding would otherwise stiffen. The multigrid
    smooths the skeleton with Chebyshev-Jacobi or, for p >= 3, Chebyshev with
    additive Schwarz on the element blocks, interpolates linearly along the edges to
    the element vertices and continues with bilinear Galerkin levels. CG keeps x and r
    in float64, the rest in float32, and one iteration is replayed as a CUDA graph.

    Args:
        nvoxels, lengths: voxel grid (divisible by s) and domain size, square voxels.
        nu: Poisson ratio, Young's modulus is the voxel coefficient.
        fixed: bool array (2, skeleton nodes), True on Dirichlet dofs, nodes ordered
            as `skeleton_nodes(nx / s, ny / s, p)`; fixed vertices must cover the
            fixed dofs for the coarse masks.
        degree, sub_voxels: p and s.
        smoothing: Chebyshev degree per level from fine to coarse (last repeats).
        lmax_safety: factor on the warm-started power iteration estimate of the
            largest eigenvalue; while members form the design changes quickly, the
            estimate lags and 1.1 let CG stall for hundreds of iterations.
        schwarz: additive Schwarz smoother on the skeleton, by default for p >= 3.
        void_floor: lower bound on the voxel coefficients seen by the solver. Float32
            condensed matrices of elements that mix solid and 10^-9 void lose the void
            stiffness to rounding, the void part then floats and CG stalls; 10^-6
            changes the compliance by about 10^-4 and restores convergence.
    """

    def __init__(
        self,
        nvoxels,
        lengths,
        nu,
        fixed,
        degree=3,
        sub_voxels=8,
        smoothing=(1, 2, 4, 8),
        coarse_dofs=300,
        eig_ratio=30.0,
        power_iters=8,
        lmax_safety=1.3,
        schwarz=None,
        void_floor=1e-6,
        chunk=1 << 15,
    ):
        NX, NY = nvoxels
        p, s = degree, sub_voxels
        assert NX % s == 0 and NY % s == 0 and p >= 2
        h = lengths[0] / NX
        assert np.isclose(h, lengths[1] / NY), "square voxels"
        ex, ey = NX // s, NY // s
        self.nvoxels, self.p, self.sub, self.ex, self.ey = (NX, NY), p, s, ex, ey
        self.E = ex * ey
        self.chunk, self.void_floor = chunk, void_floor
        self.eig_ratio, self.power_iters = eig_ratio, power_iters
        self.lmax_safety = lmax_safety
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.graph = None
        self.coarse_inverse = None
        schwarz = p >= 3 if schwarz is None else schwarz

        # element data: voxel matrices split into boundary / interior blocks
        G = voxel_matrices(p, s, h, nu)
        boundary, interior = boundary_layout(p)
        node = lambda ab: ab[0] * (p + 1) + ab[1]  # noqa: E731
        bd = np.array([2 * node(ab) + f for f in range(2) for ab in boundary])
        idofs = np.array([2 * node(ab) + f for f in range(2) for ab in interior])
        nd, ni = len(bd), len(idofs)
        self.iu, self.ju = np.triu_indices(nd)
        self.Gbb = cp.asarray(G[:, bd][:, :, bd].reshape(s * s, -1), dtype=cp.float32)
        self.Gbi = cp.asarray(
            G[:, bd][:, :, idofs].reshape(s * s, -1), dtype=cp.float32
        )
        self.Gii = cp.asarray(
            G[:, idofs][:, :, idofs].reshape(s * s, -1), dtype=cp.float32
        )
        g, wg = np.polynomial.legendre.leggauss(p + 1)
        V1, D1 = zip(
            *(lagrange(gauss_lobatto(p), -1 + (2 * k + g + 1) / s) for k in range(s))
        )
        self.V1 = cp.asarray(np.array(V1), dtype=cp.float32)
        self.D1 = cp.asarray(np.array(D1) * 2 / (s * h), dtype=cp.float32)
        self.wq = cp.asarray(wg * h / 2, dtype=cp.float32)
        self.dmat = [np.float32(v / (1 - nu**2)) for v in (1.0, nu, (1 - nu) / 2)]
        self.ne = 2 * (p + 1) ** 2
        self.full = cp.asarray(np.concatenate([bd, idofs]), dtype=cp.int32)
        self.X = cp.zeros((ni * nd, self.E), cp.float32)

        gll = gauss_lobatto(p)
        xi = np.array([gll[a] for a, _ in boundary])
        eta = np.array([gll[b] for _, b in boundary])
        self.xy = cp.asarray(np.concatenate([xi, eta]))
        self.rot = 1.0 / np.sum(xi**2 + eta**2)
        self.tk = cp.asarray((gll[1:-1] + 1) / 2, dtype=cp.float32)

        # skeleton -> element vertices: rigid-projected bilinear interpolation Q
        nb = 4 * p
        R = np.zeros((nd, 3))
        R[:nb, 0] = R[nb:, 1] = 1.0
        R[:nb, 2], R[nb:, 2] = -eta, xi
        Pi = np.eye(nd) - R @ np.linalg.solve(R.T @ R, R.T)
        T = np.zeros((nd, 8))
        for b, (x, y) in enumerate(zip(xi, eta)):
            for m in range(4):
                w = (
                    ((1 + x) if m & 1 else (1 - x))
                    * ((1 + y) if m >> 1 else (1 - y))
                    / 4
                )
                T[b, 2 * m] = T[nb + b, 2 * m + 1] = w
        self.Q = cp.asarray(Pi @ T)
        K = G.sum(0)
        X_ref = np.linalg.solve(K[np.ix_(idofs, idofs)], K[np.ix_(idofs, bd)])
        S_ref = K[np.ix_(bd, bd)] - K[np.ix_(bd, idofs)] @ X_ref
        self.reference = [
            cp.asarray(a, dtype=cp.float32)
            for a in (S_ref[self.iu, self.ju], X_ref, (Pi @ T).T @ S_ref @ (Pi @ T))
        ]

        fixed = np.asarray(fixed, dtype=bool)
        self.mask = cp.asarray(~fixed, dtype=cp.uint8).ravel()
        fine = SkeletonLevel(ex, ey, p, self.mask, schwarz, (self.xy, self.rot))
        assert fine.nn == fixed.shape[1]
        if schwarz:
            # unit block of an interior element: invert the centre of a 3 x 3 patch
            patch = SkeletonLevel(3, 3, p, None, True, (self.xy, self.rot))
            patch.mask = cp.ones(patch.ndof, cp.uint8)
            patch.S[...] = self.reference[0][:, None]
            patch.invert_blocks(cp.asarray([4], dtype=cp.int64))
            fine.B_ref = cp.ascontiguousarray(patch.B[:, 4])
            # elements touching a support or the domain boundary are always inverted
            clamped = np.zeros((ex, ey), bool)
            clamped[[0, -1], :] = clamped[:, [0, -1]] = True
            fi, fj = [a[fixed.any(0)] for a in skeleton_nodes(ex, ey, p)]
            for di in (0, 1):
                for dj in (0, 1):
                    ci = np.clip(fi // p - di * (fi % p == 0), 0, ex - 1)
                    cj = np.clip(fj // p - dj * (fj % p == 0), 0, ey - 1)
                    clamped[ci, cj] = True
            fine.clamped = cp.asarray(clamped.ravel())
        self.levels = [fine]
        nv = (ex + 1) * (ey + 1)
        mask = cp.asarray(~fixed[:, :nv].reshape(2, ex + 1, ey + 1), dtype=cp.uint8)
        level = BilinearLevel(ex, ey, mask.ravel())
        self.levels.append(level)
        while level.ex % 2 == 0 and level.ey % 2 == 0 and level.ndof > coarse_dofs:
            mask = mask[:, ::2, ::2]
            level = BilinearLevel(
                level.ex // 2, level.ey // 2, cp.ascontiguousarray(mask).ravel()
            )
            self.levels.append(level)
        kron = [np.kron(T_, T_) for T_ in bilinear_children()]
        self.G_children = cp.asarray(np.concatenate(kron), dtype=cp.float32)

        smoothing = [smoothing] if np.isscalar(smoothing) else list(smoothing)
        for i, lv in enumerate(self.levels):
            lv.smoothing = smoothing[min(i, len(smoothing) - 1)]
            lv.cheb = cp.zeros(2 * lv.smoothing - 1, cp.float32)

        self.ndof = fine.ndof
        self.x, self.r = cp.zeros(self.ndof), cp.zeros(self.ndof)
        self.p_, self.Ap = (
            cp.zeros(self.ndof, cp.float32),
            cp.zeros(self.ndof, cp.float32),
        )
        self.scalars = cp.zeros(5)
        self.partial = cp.zeros(DOT_BLOCKS)

    def update(self, coeff):
        """condense the elements and set up the hierarchy for a new voxel coefficient grid."""
        cp.cuda.get_current_stream().synchronize()
        with self.stream:
            self._update(coeff)
        self.stream.synchronize()

    def _update(self, coeff):
        fine, first = self.levels[0], self.levels[1]
        coeff = cp.maximum(
            cp.asarray(coeff, dtype=cp.float32), cp.float32(self.void_floor)
        )
        voxels = group_children(coeff, self.sub)
        cmin, cmax = voxels.min(axis=1), voxels.max(axis=1)
        args = (
            np.int64(self.E),
            cmin,
            cmax,
            *self.reference,
            fine.S,
            self.X,
            first.mats,
        )
        launch(kernel(f"scale_reference<{self.p}>"), self.E, args)
        mixed = cp.flatnonzero(cmin != cmax)
        for e0 in range(0, len(mixed), self.chunk):
            index = mixed[e0 : e0 + self.chunk]
            C = voxels[index]
            Kbb, Kbi, Kii = C @ self.Gbb, C @ self.Gbi, C @ self.Gii
            args = (np.int64(self.E), np.int32(len(C)), index, Kbb, Kbi, Kii, self.Q)
            args += (fine.S, self.X, first.mats)
            launch(kernel(f"condense<{self.p}>"), len(C), args)
        for coarse_fine, coarse in zip(self.levels[1:], self.levels[2:]):
            children = group_children(
                coarse_fine.mats.reshape(coarse_fine.ex, coarse_fine.ey, 64), 2
            )
            cp.matmul(children, self.G_children, out=coarse.mats)
        if fine.schwarz:
            fine.set_schwarz(cmin, cmax)
        for lv in self.levels[:-1]:
            lv.set_diagonal()
            hi = self.lmax_safety * power_iteration(lv, self.power_iters)
            lv.cheb[...] = cp.asarray(
                chebyshev_coefficients(hi, self.eig_ratio, lv.smoothing),
                dtype=cp.float32,
            )
        inverse = cp.linalg.inv(cp.asarray(self.levels[-1].dense()))
        if self.coarse_inverse is None:
            self.coarse_inverse = inverse
        else:
            self.coarse_inverse[...] = inverse

    def _vcycle(self):  # levels[0].b -> levels[0].x
        L = len(self.levels) - 1
        for i in range(L):
            fine, coarse = self.levels[i], self.levels[i + 1]
            fine.smooth(pre=True)
            if i == 0:
                args = (
                    np.int32(fine.ex),
                    np.int32(fine.ey),
                    fine.r,
                    coarse.b,
                    coarse.mask,
                    self.tk,
                )
                launch(kernel(f"skeleton_restrict<{self.p}, float>"), coarse.nn, args)
            else:
                args = (
                    np.int32(fine.ex),
                    np.int32(fine.ey),
                    fine.r,
                    coarse.b,
                    coarse.mask,
                )
                launch(kernel("restrict_<float>"), coarse.nn, args)
        coarse = self.levels[L]
        launch(
            kernel("dense_matvec<float>"),
            coarse.ndof,
            (self.coarse_inverse, coarse.b, coarse.x, np.int32(coarse.ndof)),
        )
        for i in reversed(range(L)):
            fine, coarse = self.levels[i], self.levels[i + 1]
            if i == 0:
                args = (
                    np.int32(fine.ex),
                    np.int32(fine.ey),
                    coarse.x,
                    fine.x,
                    fine.mask,
                    self.tk,
                )
                launch(kernel(f"skeleton_prolong<{self.p}, float>"), fine.nn, args)
            else:
                args = (
                    np.int32(fine.ex),
                    np.int32(fine.ey),
                    coarse.x,
                    fine.x,
                    fine.mask,
                )
                launch(kernel("prolong<float>"), fine.nn, args)
            fine.smooth(pre=False)

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
        x, r, p, Ap, z = self.x, self.r, self.p_, self.Ap, self.levels[0].x
        s = self.scalars
        self.levels[0].apply(p, Ap, acc="double")
        self._dot(p, Ap, s[1])
        _xr_update(p, Ap, s, x, r)
        self._dot(r, r, s[2])
        self._dot(r, z, s[3])
        self._precondition()
        self._dot(r, z, s[4])
        _p_update(z, s, p)
        s[0:1] = s[4:5]

    def _restart(self, rhs):  # true residual r = b - A x, p = z
        r, p, z = self.r, self.p_, self.levels[0].x
        self.levels[0].residual(rhs, self.x, r, acc="double")
        _mask(self.mask, r)
        self._precondition()
        p[...] = z
        self._dot(r, z, self.scalars[0])
        self._dot(r, r, self.scalars[2])

    def solve(self, rhs, x0=None, rtol=1e-8, maxiter=500, check=20):
        """flexible PCG on the skeleton; returns (u, iterations).

        u is the solver's float64 buffer, overwritten by the next solve. CG restarts
        from the true residual when its best norm drops less than 2x over `check`
        iterations.
        """
        cp.cuda.get_current_stream().synchronize()
        x = self.x
        rhs = cp.asarray(rhs, dtype=cp.float64)
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
        W = cp.empty((self.ne, self.E), cp.float32)
        args = (
            np.int32(self.ex),
            np.int32(self.ey),
            u,
            self.X,
            W,
            self.xy,
            self.rot,
            self.full,
        )
        launch(kernel(f"deformation<{self.p}>"), self.E, args)
        out = cp.empty(self.nvoxels, cp.float32)
        args = (
            np.int32(self.ex),
            np.int32(self.ey),
            np.int32(self.sub),
            self.V1,
            self.D1,
        )
        args += (self.wq, *self.dmat, W, out)
        launch(kernel(f"voxel_energy<{self.p}>"), out.size, args)
        return out
