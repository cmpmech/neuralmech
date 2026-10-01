"""matrix-free multigrid-preconditioned CG for Q_p elements over s^D voxels."""

from functools import cache

import cupy as cp
import cupyx
import numpy as np

# D = 2 grids are handled as D = 3 grids with a single node layer along the last axis
KERNELS = r"""
// one block row (threadIdx.y) per element of the given color, elements of one color
// share no nodes, so out += K_e u_e needs no atomics and sums in a fixed order
template <typename T, typename IN, typename OUT, int D, int NF, int NP, int NQ,
          bool FITTED, bool DIAG, int TX, int EPB>
__global__ void element_op(const int e0, const int e1, const int e2, const int color,
                           const int sub, const float* weights, const T* tables,
                           const T lam, const T mu, const IN* u, OUT* out)
{
    constexpr int NP2 = D == 3 ? NP : 1, NQ2 = D == 3 ? NQ : 1;
    constexpr int NN = NP * NP * NP2, NQD = NQ * NQ * NQ2, NC = NF * D;
    __shared__ T phi[NQ * NP], dphi[NQ * NP], qw[NQ], geo[4];
    constexpr int MA = D == 3 ? NP * NP * NQ : NP * NQ, MB = NP * NQ * NQ;
    constexpr int TMP = 2 * MA + (D == 3 ? 3 * MB : 0);
    __shared__ T ue[EPB][DIAG ? 1 : NF * NN];
    __shared__ T sig[EPB][DIAG ? 1 : NQD * NC];
    __shared__ T tmp[EPB][DIAG ? 1 : TMP];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    for (int k = tid; k < NQ * NP; k += TX * EPB) {
        phi[k] = tables[k];
        dphi[k] = tables[NQ * NP + k];
    }
    for (int k = tid; k < NQ; k += TX * EPB) qw[k] = tables[2 * NQ * NP + k];
    if (tid < 4) geo[tid] = tables[2 * NQ * NP + NQ + tid];

    const int b0 = color & 1, b1 = (color >> 1) & 1, b2 = (color >> 2) & 1;
    const int c0 = (e0 - b0 + 1) / 2, c1 = (e1 - b1 + 1) / 2;
    const int c2 = D == 3 ? (e2 - b2 + 1) / 2 : 1;
    const long long idx = (long long)blockIdx.x * EPB + ty;
    const bool active = idx < (long long)c0 * c1 * c2;
    const int i2 = active ? idx % c2 : 0, i1 = active ? (idx / c2) % c1 : 0;
    const int i0 = active ? idx / ((long long)c2 * c1) : 0;
    const int E0 = 2 * i0 + b0, E1 = 2 * i1 + b1, E2 = D == 3 ? 2 * i2 + b2 : 0;
    const int N1 = (NP - 1) * e1 + 1, N2 = D == 3 ? (NP - 1) * e2 + 1 : 1;
    const long long nn = (long long)((NP - 1) * e0 + 1) * N1 * N2;
    const long long elem = ((long long)E0 * e1 + E1) * (D == 3 ? e2 : 1) + E2;
    const int V1 = e1 * sub, V2 = D == 3 ? e2 * sub : 1, QPV = NQ / sub;

    auto node = [&](const int a) {
        const int a2 = a % NP2, a1 = (a / NP2) % NP, a0 = a / (NP2 * NP);
        const int g0 = (NP - 1) * E0 + a0, g1 = (NP - 1) * E1 + a1;
        const int g2 = (NP - 1) * E2 + a2;
        return ((long long)g0 * N1 + g1) * N2 + g2;
    };
    auto weight = [&](const int q, const int q0, const int q1, const int q2) {
        if (FITTED) return (T)weights[elem * NQD + q];
        const long long v = ((long long)(E0 * sub + q0 / QPV) * V1 + E1 * sub + q1 / QPV)
                            * V2 + (D == 3 ? E2 * sub + q2 / QPV : 0);
        T w = (T)weights[v] * qw[q0] * qw[q1] * geo[3];
        return D == 3 ? w * qw[q2] : w;
    };
    // gradient of local basis function a at point q
    auto grad = [&](const int a, const int q0, const int q1, const int q2, T g[3]) {
        const int a2 = a % NP2, a1 = (a / NP2) % NP, a0 = a / (NP2 * NP);
        const T v0 = phi[q0 * NP + a0], v1 = phi[q1 * NP + a1];
        const T v2 = D == 3 ? phi[q2 * NP + a2] : (T)1;
        g[0] = dphi[q0 * NP + a0] * geo[0] * v1 * v2;
        g[1] = v0 * dphi[q1 * NP + a1] * geo[1] * v2;
        g[2] = D == 3 ? v0 * v1 * dphi[q2 * NP + a2] * geo[2] : (T)0;
    };

    if (DIAG) {
        __syncthreads();
        if (!active) return;
        for (int k = tx; k < NF * NN; k += TX) {
            const int f = k / NN, a = k % NN;
            T r = 0;
            for (int q = 0; q < NQD; ++q) {
                const int q2 = q % NQ2, q1 = (q / NQ2) % NQ, q0 = q / (NQ2 * NQ);
                T g[3];
                grad(a, q0, q1, q2, g);
                T gg = 0;
                #pragma unroll
                for (int d = 0; d < D; ++d) gg += g[d] * g[d];
                const T gf = g[f % D];
                r += weight(q, q0, q1, q2)
                     * (NF == 1 ? gg : lam * gf * gf + mu * (gg + gf * gf));
            }
            out[(long long)f * nn + node(a)] += (OUT)r;
        }
        return;
    }

    // sum factorization: gradients at the points one direction at a time
    T* U = ue[ty];
    T* S = sig[ty];
    T* A = tmp[ty];
    T* B = tmp[ty] + 2 * MA;
    if (active)
        for (int k = tx; k < NF * NN; k += TX)
            U[k] = (T)u[(k / NN) * nn + node(k % NN)];
    __syncthreads();

    for (int f = 0; f < NF; ++f) {
        const T* uf = U + f * NN;
        if (D == 3) {
            if (active)
                for (int k = tx; k < MA; k += TX) {  // (a0, a1, q2)
                    const int q2 = k % NQ, a1 = (k / NQ) % NP, a0 = k / (NQ * NP);
                    T v = 0, w = 0;
                    for (int a2 = 0; a2 < NP; ++a2) {
                        const T x = uf[(a0 * NP + a1) * NP + a2];
                        v += phi[q2 * NP + a2] * x;
                        w += dphi[q2 * NP + a2] * x;
                    }
                    A[k] = v;
                    A[MA + k] = w;
                }
            __syncthreads();
            if (active)
                for (int k = tx; k < MB; k += TX) {  // (a0, q1, q2)
                    const int q2 = k % NQ, q1 = (k / NQ) % NQ, a0 = k / (NQ * NQ);
                    T vv = 0, dv = 0, vd = 0;
                    for (int a1 = 0; a1 < NP; ++a1) {
                        const int m = (a0 * NP + a1) * NQ + q2;
                        vv += phi[q1 * NP + a1] * A[m];
                        dv += dphi[q1 * NP + a1] * A[m];
                        vd += phi[q1 * NP + a1] * A[MA + m];
                    }
                    B[k] = vv;
                    B[MB + k] = dv;
                    B[2 * MB + k] = vd;
                }
            __syncthreads();
            if (active)
                for (int q = tx; q < NQD; q += TX) {
                    const int r12 = q % (NQ * NQ), q0 = q / (NQ * NQ);
                    T g0 = 0, g1 = 0, g2 = 0;
                    for (int a0 = 0; a0 < NP; ++a0) {
                        const int m = a0 * NQ * NQ + r12;
                        g0 += dphi[q0 * NP + a0] * B[m];
                        g1 += phi[q0 * NP + a0] * B[MB + m];
                        g2 += phi[q0 * NP + a0] * B[2 * MB + m];
                    }
                    S[q * NC + f * D] = g0 * geo[0];
                    S[q * NC + f * D + 1] = g1 * geo[1];
                    S[q * NC + f * D + 2] = g2 * geo[2];
                }
        } else {
            if (active)
                for (int k = tx; k < MA; k += TX) {  // (a0, q1)
                    const int q1 = k % NQ, a0 = k / NQ;
                    T v = 0, w = 0;
                    for (int a1 = 0; a1 < NP; ++a1) {
                        v += phi[q1 * NP + a1] * uf[a0 * NP + a1];
                        w += dphi[q1 * NP + a1] * uf[a0 * NP + a1];
                    }
                    A[k] = v;
                    A[MA + k] = w;
                }
            __syncthreads();
            if (active)
                for (int q = tx; q < NQD; q += TX) {
                    const int q1 = q % NQ, q0 = q / NQ;
                    T g0 = 0, g1 = 0;
                    for (int a0 = 0; a0 < NP; ++a0) {
                        g0 += dphi[q0 * NP + a0] * A[a0 * NQ + q1];
                        g1 += phi[q0 * NP + a0] * A[MA + a0 * NQ + q1];
                    }
                    S[q * NC + f * D] = g0 * geo[0];
                    S[q * NC + f * D + 1] = g1 * geo[1];
                }
        }
        __syncthreads();
    }

    // stress at the points, scaled by the weight and the inverse element size
    if (active)
        for (int q = tx; q < NQD; q += TX) {
            const int q2 = q % NQ2, q1 = (q / NQ2) % NQ, q0 = q / (NQ2 * NQ);
            const T w = weight(q, q0, q1, q2);
            T* s = S + q * NC;
            if (NF == 1) {
                #pragma unroll
                for (int d = 0; d < D; ++d) s[d] *= w * geo[d];
            } else {
                T gu[NF][D];
                T tr = 0;
                #pragma unroll
                for (int f = 0; f < NF; ++f)
                    #pragma unroll
                    for (int d = 0; d < D; ++d) gu[f][d] = s[f * D + d];
                #pragma unroll
                for (int d = 0; d < D; ++d) tr += gu[d % NF][d];
                #pragma unroll
                for (int f = 0; f < NF; ++f)
                    #pragma unroll
                    for (int d = 0; d < D; ++d)
                        s[f * D + d] = w * geo[d] * ((f == d ? lam * tr : (T)0)
                                        + mu * (gu[f][d] + gu[d % NF][f % D]));
            }
        }
    __syncthreads();

    // transposed sum factorization back to the nodes
    for (int f = 0; f < NF; ++f) {
        if (D == 3) {
            if (active)
                for (int k = tx; k < MB; k += TX) {  // (a0, q1, q2)
                    const int r12 = k % (NQ * NQ), a0 = k / (NQ * NQ);
                    T x0 = 0, x1 = 0, x2 = 0;
                    for (int q0 = 0; q0 < NQ; ++q0) {
                        const T* s = S + (q0 * NQ * NQ + r12) * NC + f * D;
                        x0 += dphi[q0 * NP + a0] * s[0];
                        x1 += phi[q0 * NP + a0] * s[1];
                        x2 += phi[q0 * NP + a0] * s[2];
                    }
                    B[k] = x0;
                    B[MB + k] = x1;
                    B[2 * MB + k] = x2;
                }
            __syncthreads();
            if (active)
                for (int k = tx; k < MA; k += TX) {  // (a0, a1, q2)
                    const int q2 = k % NQ, a1 = (k / NQ) % NP, a0 = k / (NQ * NP);
                    T v = 0, w = 0;
                    for (int q1 = 0; q1 < NQ; ++q1) {
                        const int m = (a0 * NQ + q1) * NQ + q2;
                        v += phi[q1 * NP + a1] * B[m] + dphi[q1 * NP + a1] * B[MB + m];
                        w += phi[q1 * NP + a1] * B[2 * MB + m];
                    }
                    A[k] = v;
                    A[MA + k] = w;
                }
            __syncthreads();
            if (active)
                for (int a = tx; a < NN; a += TX) {
                    const int a2 = a % NP, a1 = (a / NP) % NP, a0 = a / (NP * NP);
                    T r = 0;
                    for (int q2 = 0; q2 < NQ; ++q2) {
                        const int m = (a0 * NP + a1) * NQ + q2;
                        r += phi[q2 * NP + a2] * A[m] + dphi[q2 * NP + a2] * A[MA + m];
                    }
                    out[(long long)f * nn + node(a)] += (OUT)r;
                }
        } else {
            if (active)
                for (int k = tx; k < MA; k += TX) {  // (a0, q1)
                    const int q1 = k % NQ, a0 = k / NQ;
                    T x0 = 0, x1 = 0;
                    for (int q0 = 0; q0 < NQ; ++q0) {
                        const T* s = S + (q0 * NQ + q1) * NC + f * D;
                        x0 += dphi[q0 * NP + a0] * s[0];
                        x1 += phi[q0 * NP + a0] * s[1];
                    }
                    A[k] = x0;
                    A[MA + k] = x1;
                }
            __syncthreads();
            if (active)
                for (int a = tx; a < NN; a += TX) {
                    const int a1 = a % NP, a0 = a / NP;
                    T r = 0;
                    for (int q1 = 0; q1 < NQ; ++q1)
                        r += phi[q1 * NP + a1] * A[a0 * NQ + q1]
                             + dphi[q1 * NP + a1] * A[MA + a0 * NQ + q1];
                    out[(long long)f * nn + node(a)] += (OUT)r;
                }
        }
        __syncthreads();
    }
}

// fine node g lies in coarse element e = g / R at local position xi[g % R], xi[R] = 1
__device__ __forceinline__ void coarse_of(const int g, const int nc, const int R, int& e,
                                          int& j)
{
    e = g / R;
    j = g % R;
    if (e >= nc - 1) {
        e = nc - 2;
        j = g - e * R;
    }
}

// xf += mask (P xc), tensor-product linear interpolation from the coarse nodes
template <typename T>
__global__ void prolong(const int f0, const int f1, const int f2, const int c0,
                        const int c1, const int c2, const int R, const int nf,
                        const T* xi, const T* xc, T* xf, const unsigned char* mask)
{
    const long long nn = (long long)f0 * f1 * f2, nnc = (long long)c0 * c1 * c2;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nn) return;
    const int g[3] = {(int)(t / ((long long)f1 * f2)), (int)((t / f2) % f1),
                      (int)(t % f2)};
    const int nc[3] = {c0, c1, c2};
    int e[3], lo[3];
    T w[3][2];
    for (int d = 0; d < 3; ++d) {
        if (nc[d] == 1) {
            e[d] = 0;
            lo[d] = 1;
            w[d][0] = 1;
            w[d][1] = 0;
            continue;
        }
        int j;
        coarse_of(g[d], nc[d], R, e[d], j);
        lo[d] = 2;
        w[d][0] = 1 - xi[j];
        w[d][1] = xi[j];
    }
    for (int f = 0; f < nf; ++f) {
        T v = 0;
        for (int a = 0; a < lo[0]; ++a)
            for (int b = 0; b < lo[1]; ++b)
                for (int c = 0; c < lo[2]; ++c)
                    v += w[0][a] * w[1][b] * w[2][c]
                         * xc[f * nnc + ((long long)(e[0] + a) * c1 + e[1] + b) * c2 + e[2] + c];
        const long long k = f * nn + t;
        xf[k] += mask[k] ? v : (T)0;
    }
}

// bc = mask (P^T rf)
template <typename T>
__global__ void restrict_(const int f0, const int f1, const int f2, const int c0,
                          const int c1, const int c2, const int R, const int nf,
                          const T* xi, const T* rf, T* bc, const unsigned char* mask)
{
    const long long nn = (long long)f0 * f1 * f2, nnc = (long long)c0 * c1 * c2;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nnc) return;
    const int I[3] = {(int)(t / ((long long)c1 * c2)), (int)((t / c2) % c1),
                      (int)(t % c2)};
    const int nc[3] = {c0, c1, c2}, nfd[3] = {f0, f1, f2};
    int lo[3], hi[3];
    for (int d = 0; d < 3; ++d) {
        lo[d] = nc[d] == 1 ? 0 : max(0, (I[d] - 1) * R + 1);
        hi[d] = nc[d] == 1 ? 0 : min(nfd[d] - 1, (I[d] + 1) * R - 1);
    }
    auto wt = [&](const int d, const int g) -> T {
        if (nc[d] == 1) return 1;
        int e, j;
        coarse_of(g, nc[d], R, e, j);
        return e == I[d] ? 1 - xi[j] : (e + 1 == I[d] ? xi[j] : (T)0);
    };
    for (int f = 0; f < nf; ++f) {
        T s = 0;
        for (int a = lo[0]; a <= hi[0]; ++a) {
            const T wa = wt(0, a);
            for (int b = lo[1]; b <= hi[1]; ++b) {
                const T wb = wa * wt(1, b);
                for (int c = lo[2]; c <= hi[2]; ++c)
                    s += wb * wt(2, c) * rf[f * nn + ((long long)a * f1 + b) * f2 + c];
            }
        }
        const long long k = f * nnc + t;
        bc[k] = mask[k] ? s : (T)0;
    }
}

// unit-coefficient strain energy of every voxel, NG^D Gauss points per voxel, one
// block per element; element translations are removed in float64 before float32 math
template <int D, int NF, int NP, int NG, int S>
__global__ void voxel_energy(const int e0, const int e1, const int e2, const float* tables,
                             const float lam, const float mu, const double* u,
                             float* energy)
{
    constexpr int NP2 = D == 3 ? NP : 1, NG2 = D == 3 ? NG : 1, S2 = D == 3 ? S : 1;
    constexpr int NN = NP * NP * NP2, NQ = S * NG;
    __shared__ float phi[NQ * NP], dphi[NQ * NP], gw[NG], geo[4], ue[NF * NN];
    __shared__ double u0[NF];
    const long long elem = blockIdx.x;
    const int c2 = D == 3 ? e2 : 1;
    const int E2 = elem % c2, E1 = (elem / c2) % e1, E0 = elem / ((long long)c2 * e1);
    const int N1 = (NP - 1) * e1 + 1, N2 = D == 3 ? (NP - 1) * e2 + 1 : 1;
    const long long nn = (long long)((NP - 1) * e0 + 1) * N1 * N2;
    auto node = [&](const int a) {
        const int a2 = a % NP2, a1 = (a / NP2) % NP, a0 = a / (NP2 * NP);
        return ((long long)((NP - 1) * E0 + a0) * N1 + (NP - 1) * E1 + a1) * N2
               + (NP - 1) * E2 + a2;
    };
    for (int k = threadIdx.x; k < NQ * NP; k += blockDim.x) {
        phi[k] = tables[k];
        dphi[k] = tables[NQ * NP + k];
    }
    for (int k = threadIdx.x; k < NG; k += blockDim.x) gw[k] = tables[2 * NQ * NP + k];
    if (threadIdx.x < 4) geo[threadIdx.x] = tables[2 * NQ * NP + NG + threadIdx.x];
    if (threadIdx.x < NF) u0[threadIdx.x] = u[threadIdx.x * nn + node(0)];
    __syncthreads();
    for (int k = threadIdx.x; k < NF * NN; k += blockDim.x)
        ue[k] = (float)(u[(k / NN) * nn + node(k % NN)] - u0[k / NN]);
    __syncthreads();

    const int V1 = e1 * S, V2 = D == 3 ? e2 * S : 1;
    for (int l = threadIdx.x; l < S * S * S2; l += blockDim.x) {
        const int l2 = l % S2, l1 = (l / S2) % S, l0 = l / (S2 * S);
        float acc = 0;
        for (int g = 0; g < NG * NG * NG2; ++g) {
            const int g2 = g % NG2, g1 = (g / NG2) % NG, g0 = g / (NG2 * NG);
            const int q0 = l0 * NG + g0, q1 = l1 * NG + g1, q2 = l2 * NG + g2;
            float G[NF][3] = {};
            for (int a = 0; a < NN; ++a) {
                const int a2 = a % NP2, a1 = (a / NP2) % NP, a0 = a / (NP2 * NP);
                const float v0 = phi[q0 * NP + a0], v1 = phi[q1 * NP + a1];
                const float v2 = D == 3 ? phi[q2 * NP + a2] : 1.0f;
                const float d0 = dphi[q0 * NP + a0] * geo[0] * v1 * v2;
                const float d1 = v0 * dphi[q1 * NP + a1] * geo[1] * v2;
                const float d2 = D == 3 ? v0 * v1 * dphi[q2 * NP + a2] * geo[2] : 0.0f;
                #pragma unroll
                for (int f = 0; f < NF; ++f) {
                    const float x = ue[f * NN + a];
                    G[f][0] += x * d0;
                    G[f][1] += x * d1;
                    G[f][2] += x * d2;
                }
            }
            float w = gw[g0] * gw[g1] * (D == 3 ? gw[g2] : 1.0f), e = 0;
            if (NF == 1) {
                #pragma unroll
                for (int d = 0; d < D; ++d) e += G[0][d] * G[0][d];
            } else {
                float tr = 0;
                #pragma unroll
                for (int d = 0; d < D; ++d) tr += G[d % NF][d];
                e = lam * tr * tr;
                #pragma unroll
                for (int f = 0; f < NF; ++f)
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        const float eps = 0.5f * (G[f][d] + G[d % NF][f % D]);
                        e += 2.0f * mu * eps * eps;
                    }
            }
            acc += w * e;
        }
        const long long v = ((long long)(E0 * S + l0) * V1 + E1 * S + l1) * V2
                            + (D == 3 ? E2 * S + l2 : 0);
        energy[v] = acc * geo[3];
    }
}

// w_q = int_e c l_q from the voxels of an element, one block per element and a thread
// per point; L holds the integrals of the Lagrange polynomials over each voxel
template <int D, int NQ>
__global__ void fit_voxels(const int e1, const int e2, const int sub, const float* L,
                           const float volume, const float* c, float* w)
{
    constexpr int NQ2 = D == 3 ? NQ : 1, NQD = NQ * NQ * NQ2;
    __shared__ float Ls[64 * NQ];
    for (int k = threadIdx.x; k < sub * NQ; k += blockDim.x) Ls[k] = L[k];
    __syncthreads();
    const long long elem = blockIdx.x;
    const int c2 = D == 3 ? e2 : 1;
    const int E2 = elem % c2, E1 = (elem / c2) % e1, E0 = elem / ((long long)c2 * e1);
    const int V1 = e1 * sub, V2 = D == 3 ? e2 * sub : 1, sub2 = D == 3 ? sub : 1;
    for (int q = threadIdx.x; q < NQD; q += blockDim.x) {
        const int q2 = q % NQ2, q1 = (q / NQ2) % NQ, q0 = q / (NQ2 * NQ);
        float acc = 0;
        for (int l0 = 0; l0 < sub; ++l0)
            for (int l1 = 0; l1 < sub; ++l1) {
                const float w01 = Ls[l0 * NQ + q0] * Ls[l1 * NQ + q1];
                const long long row = ((long long)(E0 * sub + l0) * V1 + E1 * sub + l1) * V2
                                      + (D == 3 ? E2 * sub : 0);
                for (int l2 = 0; l2 < sub2; ++l2)
                    acc += c[row + l2] * w01 * (D == 3 ? Ls[l2 * NQ + q2] : 1.0f);
            }
        w[elem * NQD + q] = acc * volume;
    }
}

// every voxel of an element gains alpha times the P-norm of the element's coefficients,
// dfloor = d(raised coefficient of any voxel of the element) / d(voxel coefficient)
template <int D, int P>
__global__ void element_floor(const int e1, const int e2, const int sub, const float alpha,
                              const float* c, float* out, float* dfloor)
{
    __shared__ float red[256];
    const long long elem = blockIdx.x;
    const int c2 = D == 3 ? e2 : 1;
    const int E2 = elem % c2, E1 = (elem / c2) % e1, E0 = elem / ((long long)c2 * e1);
    const int V1 = e1 * sub, V2 = D == 3 ? e2 * sub : 1, sub2 = D == 3 ? sub : 1;
    const int n = sub * sub * sub2;
    auto voxel = [&](const int l) {
        const int l2 = l % sub2, l1 = (l / sub2) % sub, l0 = l / (sub2 * sub);
        return ((long long)(E0 * sub + l0) * V1 + E1 * sub + l1) * V2
               + (D == 3 ? E2 * sub + l2 : 0);
    };
    auto reduce = [&](float v, const bool maximum) {
        red[threadIdx.x] = v;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                const float o = red[threadIdx.x + stride];
                red[threadIdx.x] = maximum ? fmaxf(red[threadIdx.x], o) : red[threadIdx.x] + o;
            }
            __syncthreads();
        }
        const float r = red[0];
        __syncthreads();
        return r;
    };
    float m = 0;
    for (int l = threadIdx.x; l < n; l += blockDim.x) m = fmaxf(m, c[voxel(l)]);
    m = reduce(m, true);
    float s = 0;  // scaled by the maximum, the P-th powers stay in float range
    for (int l = threadIdx.x; l < n; l += blockDim.x) {
        const float r = m > 0 ? c[voxel(l)] / m : 0.0f;
        float rp = 1;
        #pragma unroll
        for (int k = 0; k < P; ++k) rp *= r;
        s += rp;
    }
    const float top = m * powf(reduce(s, false) / n, 1.0f / P);
    for (int l = threadIdx.x; l < n; l += blockDim.x) {
        const long long v = voxel(l);
        const float r = top > 0 ? c[v] / top : 0.0f;
        float rp = 1;
        #pragma unroll
        for (int k = 0; k < P - 1; ++k) rp *= r;
        out[v] = c[v] + alpha * top;
        dfloor[v] = alpha * rp / n;
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
FLOOR_POWER = 8  # the element p-norm of the floor, close to its maximum
_functions = {}


def kernel(name):
    """compiled kernel or template instance, e.g. `kernel("prolong<float>")`."""
    if name not in _functions:
        module = cp.RawModule(
            code=KERNELS, name_expressions=[name], options=("-std=c++17",)
        )
        _functions[name] = module.get_function(name)
    return _functions[name]


def launch(function, threads, args):
    function(((threads + 255) // 256,), (256,), args)


def ctype(dtype):
    return "float" if dtype == cp.float32 else "double"


_cast = cp.ElementwiseKernel("S a", "T b", "b = (T)a", "cast")
_mask = cp.ElementwiseKernel("uint8 m", "T x", "x = m ? x : (T)0", "mask")
_identity_rows = cp.ElementwiseKernel(
    "uint8 m, S u", "T out", "out = m ? out : (T)u", "identity_rows"
)
_unit_rows = cp.ElementwiseKernel(
    "uint8 m", "T out", "out = m ? out : (T)1", "unit_rows"
)
_cheb_pre = cp.ElementwiseKernel(
    "T b, T dinv, raw T cheb",
    "T r, T d, T x",
    "r = b; d = dinv * b * cheb[0]; x = d",
    "cheb_pre",
)
_cheb_post = cp.ElementwiseKernel(
    "T b, T Ad, T dinv, raw T cheb",
    "T r, T d, T x",
    "r = b - Ad; d = dinv * r * cheb[0]; x += d",
    "cheb_post",
)
_cheb_step = cp.ElementwiseKernel(
    "T Ad, T dinv, raw T cheb",
    "T r, T d, T x",
    "r -= Ad; d = cheb[0] * d + cheb[1] * dinv * r; x += d",
    "cheb_step",
)
_subtract = cp.ElementwiseKernel("T a", "T x", "x -= a", "subtract")
_add_product = cp.ElementwiseKernel("T a, T b", "T x", "x += a * b", "add_product")
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
    "F p, A Ap, raw D s",
    "D x, D r",
    "D alpha = s[0] / s[1]; x += alpha * p; r -= alpha * Ap",
    "xr_update",
)
_p_update = cp.ElementwiseKernel(
    "F z, raw D s", "F p", "p = z + (s[4] - s[3]) / s[0] * p", "p_update"
)


def gauss_lobatto(p):
    """p + 1 Gauss-Lobatto points on [0, 1]."""
    c = np.zeros(p + 1)
    c[p] = 1.0
    inner = np.polynomial.legendre.legroots(np.polynomial.legendre.legder(c))
    return 0.5 * (np.concatenate([[-1.0], np.sort(inner), [1.0]]) + 1.0)


def gauss(n, a=0.0, b=1.0):
    """n-point Gauss rule on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return a + 0.5 * (b - a) * (x + 1.0), 0.5 * (b - a) * w


def lagrange(nodes, x):
    """values and derivatives (len(x), len(nodes)) of the Lagrange polynomials."""
    x = np.asarray(x, dtype=np.float64)
    V = np.ones((len(x), len(nodes)))
    dV = np.zeros((len(x), len(nodes)))
    for a, xa in enumerate(nodes):
        others = np.delete(nodes, a)
        denom = np.prod(xa - others)
        V[:, a] = np.prod(x[:, None] - others, axis=1) / denom
        for c in range(len(others)):
            dV[:, a] += np.prod(x[:, None] - np.delete(others, c), axis=1) / denom
    return V, dV


@cache
def voxel_table(nq, sub):
    return cp.asarray(voxel_integrals(gauss(nq)[0], sub), cp.float32)


def voxel_integrals(points, sub):
    """(sub, len(points)) integrals of the Lagrange polynomials over each voxel."""
    L = np.zeros((sub, len(points)))
    for k in range(sub):
        x, w = gauss(len(points), k / sub, (k + 1) / sub)
        L[k] = w @ lagrange(points, x)[0]
    return L


def contract_weights(W, elements, R, nq_child, nq):
    """(elements, nq^D) weights of elements made of R^D children with weights W.

    Child weights integrate Q_{nq_child - 1} exactly, which contains the parent's
    Lagrange polynomials for nq <= nq_child, so this is exact (R = 1: fewer points).
    """
    D = len(elements)
    x = (np.arange(R)[:, None] + gauss(nq_child)[0][None, :]) / R
    T = cp.asarray(lagrange(gauss(nq)[0], x.ravel())[0].reshape(R, nq_child, nq))
    W = W.reshape(*[x for e in elements for x in (e, R)], *[nq_child] * D)
    if D == 2:
        W = cp.einsum("iajbpq,apx,bqy->ijxy", W.astype(cp.float64), T, T)
    else:
        W = cp.einsum("iajbkcpqr,apx,bqy,crz->ijkxyz", W.astype(cp.float64), T, T, T)
    return W.reshape(*elements, -1).astype(cp.float32)


def point_gradients(phi, dphi, invh):
    """(NQ^D, D, NP^D) basis gradients at the tensor points, both indices in C order."""
    D = len(invh)
    G = []
    for d in range(D):
        g = np.ones((1, 1))
        for k in range(D):
            g = np.kron(g, dphi * invh[k] if k == d else phi)
        G.append(g)
    return np.stack(G, axis=1)


def local_nodes(P, D):
    """(D, (P + 1)^D) local node indices of an element in C order."""
    return np.stack(np.meshgrid(*[np.arange(P + 1)] * D, indexing="ij")).reshape(D, -1)


def material_tensor(D, NF, lam, mu):
    """(NF, D, NF, D) C with stress_fd = C_fdgk grad_gk."""
    if NF == 1:
        return np.eye(D).reshape(1, D, 1, D)
    eye = np.eye(D)
    return (
        lam * np.einsum("fd,gk->fdgk", eye, eye)
        + mu * np.einsum("fg,dk->fdgk", eye, eye)
        + mu * np.einsum("fk,dg->fdgk", eye, eye)
    )


class Level:
    """one grid of the hierarchy: Q_P elements with weights at NQ^D points per element.

    Fitted levels read one weight per element point, voxel levels (the reference, one
    element per voxel) read the voxel coefficients and scale Gauss weights.
    """

    def __init__(self, solver, elements, P, NQ, fitted, sub, mask):
        self.D, self.NF = solver.D, solver.NF
        self.elements, self.P, self.NQ, self.fitted, self.sub = (
            elements,
            P,
            NQ,
            fitted,
            sub,
        )
        self.nodes = tuple(P * e + 1 for e in elements)
        self.nn = int(np.prod(self.nodes))
        self.ndof = self.NF * self.nn
        self.mask = mask
        self.h = [L / e for L, e in zip(solver.lengths, elements)]
        self.lam, self.mu = solver.lam, solver.mu

        points, qw = gauss(NQ // sub) if not fitted else (gauss(NQ)[0], None)
        if not fitted:  # NQ / sub Gauss points in each voxel
            points = np.concatenate([(k + points) / sub for k in range(sub)])
            qw = np.tile(qw, sub) / sub
        self.phi, self.dphi = lagrange(gauss_lobatto(P), points)
        self.invh = [1.0 / h for h in self.h]
        geo = self.invh + [0.0] * (3 - self.D) + [np.prod(self.h)]
        tables = np.concatenate(
            [
                self.phi.ravel(),
                self.dphi.ravel(),
                qw if qw is not None else [0] * NQ,
                geo,
            ]
        )
        self.tables = {t: cp.asarray(tables, dtype=t) for t in (cp.float32, cp.float64)}
        self.weights = None

        NN, NQD = (P + 1) ** self.D, NQ**self.D
        self.TX = min(128, -(-max(NQD, self.NF * NN) // 8) * 8)
        tmp = 2 * (P + 1) ** (self.D - 1) * NQ + (
            3 * (P + 1) * NQ**2 if self.D == 3 else 0
        )
        per_element = 8 * (self.NF * NN + NQD * self.NF * self.D + tmp)
        self.EPB = max(1, min(256 // self.TX, 40000 // per_element))
        e3 = list(elements) + [1] * (3 - self.D)
        self.colors = []
        for color in range(2**self.D):
            count = np.prod([(e3[d] - ((color >> d) & 1) + 1) // 2 for d in range(3)])
            self.colors.append((color, int(count)))

        self.b, self.x, self.r, self.d, self.Ad = (
            cp.zeros(self.ndof, cp.float32) for _ in range(5)
        )
        self.dinv = cp.ones(self.ndof, cp.float32)
        self._power_vector = None
        self._dense = None

    def op(self, u, out, T=cp.float32, diag=False):
        """out = A u with fixed dofs as identity rows (diag: the diagonal of A)."""
        e3 = list(self.elements) + [1] * (3 - self.D)
        name = (
            f"element_op<{ctype(T)}, {ctype(u.dtype)}, {ctype(out.dtype)}, {self.D}, "
            f"{self.NF}, {self.P + 1}, {self.NQ}, {str(self.fitted).lower()}, "
            f"{str(diag).lower()}, {self.TX}, {self.EPB}>"
        )
        function = kernel(name)
        out.fill(0)
        for color, count in self.colors:
            if count == 0:
                continue
            args = (*[np.int32(e) for e in e3], np.int32(color), np.int32(self.sub))
            args += (self.weights, self.tables[T], T(self.lam), T(self.mu), u, out)
            function(((count + self.EPB - 1) // self.EPB,), (self.TX, self.EPB), args)
        if diag:
            _unit_rows(self.mask, out)
        else:
            _identity_rows(self.mask, u, out)
        return out

    def set_diagonal(self):
        self.op(self.Ad, self.Ad, diag=True)
        cp.divide(1.0, self.Ad, out=self.dinv)

    def element_dofs(self, coords):
        """(len, n) global dofs of the elements at coords (D, len), local dofs f NP^D + a."""
        local = cp.asarray(local_nodes(self.P, self.D))
        g = [self.P * coords[d][:, None] + local[d][None, :] for d in range(self.D)]
        nodes = cp.ravel_multi_index(tuple(g), self.nodes)
        return cp.concatenate([nodes + f * self.nn for f in range(self.NF)], axis=1)

    def estimate_lmax(self, iters):  # power iteration on M^-1 A, from a fixed random start
        # not warm-started: over hundreds of design updates the carried vector locks
        # onto a localized mode and underestimated lmax by up to 37%, the Chebyshev
        # smoother then amplified the top modes and CG ran into maxiter
        v, Av = self.d, self.Ad
        if self._power_vector is None:
            rng = cp.random.default_rng(0)
            self._power_vector = rng.random(self.ndof, dtype=cp.float32)
        v[...] = self._power_vector
        _mask(self.mask, v)
        for _ in range(iters):
            v *= cp.dot(v, v) ** -0.5
            cp.multiply(self.dinv, self.op(v, Av), out=v)
        return float(cp.dot(v, v)) ** 0.5

    def point_matrices(self):
        """(NQ^D, n, n) element matrix contributions of every point for unit weight."""
        G = point_gradients(self.phi, self.dphi, self.invh)
        C = material_tensor(self.D, self.NF, self.lam, self.mu)
        n = self.NF * G.shape[2]
        return np.einsum("qda,fdgk,qkb->qfagb", G, C, G).reshape(len(G), n, n)

    def point_weights(self):
        """(elements, NQ^D) float64 weights of every element point."""
        if self.fitted:
            return self.weights.reshape(-1, self.NQ**self.D).astype(cp.float64)
        c = cp.asnumpy(self.weights).astype(np.float64)
        s, q = self.sub, self.NQ // self.sub
        W = c
        for d in range(self.D):  # repeat each voxel value over its Gauss points
            W = np.repeat(W, q, axis=d)
        e = self.elements
        W = W.reshape([x for k in range(self.D) for x in (e[k], self.NQ)])
        W = W.transpose(list(range(0, 2 * self.D, 2)) + list(range(1, 2 * self.D, 2)))
        qw = np.tile(gauss(q)[1], s) / s
        w = np.ones(1)
        for _ in range(self.D):
            w = np.kron(w, qw)
        return W.reshape(-1, self.NQ**self.D) * w * np.prod(self.h)

    def dense(self):
        """float64 dense matrix on the GPU with fixed dofs as identity rows."""
        if self._dense is None:
            coords = cp.stack(
                cp.unravel_index(cp.arange(int(np.prod(self.elements))), self.elements)
            )
            efts = self.element_dofs(coords)
            Kq = self.point_matrices()
            self._dense = (efts, cp.asarray(Kq.reshape(len(Kq), -1)), Kq.shape[1:])
        efts, Kq, shape = self._dense
        K_e = (cp.asarray(self.point_weights()) @ Kq).reshape(-1, *shape)
        A = cp.zeros((self.ndof, self.ndof))
        cupyx.scatter_add(A, (efts[:, :, None], efts[:, None, :]), K_e)
        m = self.mask.astype(cp.float64)
        A = m[:, None] * A * m[None, :]
        A[cp.arange(self.ndof), cp.arange(self.ndof)] += 1.0 - m
        return A


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


class VoxelMultigrid:
    """multigrid-preconditioned CG for linear elasticity or Poisson on a voxel grid.

    Q_p Lagrange elements (Gauss-Lobatto nodes) span s^D voxels. With voxelwise
    constant coefficients the element matrices only depend on the weights
    w_q = int_e c l_q of a (2p + 1)^D Gauss grid, which integrate them exactly and are
    all a fine level stores; with s = 1 the voxel coefficients are read directly. The
    hierarchy goes Q_p -> Q_1 on the same mesh, then coarsens the mesh by 2 (or 3, 5),
    every level matrix-free from its own exactly fitted weights, down to a dense coarse
    inverse.
    CG keeps x and r in float64, the rest in float32 with Chebyshev smoothing, and
    one CG iteration is replayed as a single CUDA graph.

    Args:
        nvoxels, lengths: voxel grid (divisible by s times a power of two) and domain.
        physics: "elasticity" (Young's modulus per voxel, plane stress in 2D) or
            "poisson" (conductivity per voxel).
        fixed: bool array (NF, *nodes), True on Dirichlet dofs (see `node_coordinates`).
        degree, sub_voxels: p and s.
        fitted: fitted weights on the fine level, by default for s > 1.
        smoothing: Chebyshev degree per level from fine to coarse (last repeats).
        floor: element-wise finite cell alpha, every voxel of an element gains
            `floor` times the 8-norm of the element's coefficients (close to its
            maximum, but differentiable); keeps Jacobi smoothing robust for cut
            elements with p > 1 (0 for the unmodified material).
    """

    def __init__(
        self,
        nvoxels,
        lengths,
        physics,
        fixed,
        degree=1,
        sub_voxels=1,
        nu=0.3,
        fitted=None,
        smoothing=(2, 2, 4, 8),
        floor=1e-3,
        coarse_dofs=2000,
        eig_ratio=30.0,
        safety=1.1,
        power_iters=24,
    ):
        self.D = len(nvoxels)
        self.NF = self.D if physics == "elasticity" else 1
        self.lengths, self.nvoxels = list(lengths), tuple(nvoxels)
        self.sub, self.degree = sub_voxels, degree
        if self.D == 2:
            self.lam, self.mu = nu / (1 - nu**2), 0.5 / (1 + nu)
        else:
            self.lam, self.mu = nu / ((1 + nu) * (1 - 2 * nu)), 0.5 / (1 + nu)
        self.eig_ratio, self.safety, self.power_iters = eig_ratio, safety, power_iters
        self.floor = floor
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.graph = None
        fitted = sub_voxels > 1 if fitted is None else fitted
        assert all(n % sub_voxels == 0 for n in nvoxels)

        mask = cp.asarray(~np.asarray(fixed), dtype=cp.uint8)
        elements = tuple(n // sub_voxels for n in nvoxels)
        nq = 2 * degree + 1 if fitted else sub_voxels * (degree + 1)
        self.levels = [
            Level(self, elements, degree, nq, fitted, sub_voxels, mask.ravel())
        ]
        self.transfers = []
        if degree > 1:
            mask = mask[(slice(None),) + (slice(None, None, degree),) * self.D]
            self.levels.append(Level(self, elements, 1, 3, True, 1, mask.ravel()))
            xi = np.append(gauss_lobatto(degree)[:-1], 1.0)
            self.transfers.append((degree, cp.asarray(xi, dtype=cp.float32)))
        while self.levels[-1].ndof > coarse_dofs:
            fine = self.levels[-1]
            # halve the mesh, or divide it by 3 or 5 where 2 does not divide it
            factors = [R for R in (2, 3, 5) if all(e % R == 0 for e in fine.elements)]
            if not factors:
                break
            R = factors[0]
            mask = mask[(slice(None),) + (slice(None, None, R),) * self.D]
            elements = tuple(e // R for e in fine.elements)
            self.levels.append(Level(self, elements, 1, 3, True, 1, mask.ravel()))
            xi = np.linspace(0.0, 1.0, R + 1)
            self.transfers.append((R, cp.asarray(xi, dtype=cp.float32)))
        assert self.levels[-1].ndof <= 4 * coarse_dofs, "grid does not coarsen enough"

        smoothing = [smoothing] if np.isscalar(smoothing) else list(smoothing)
        for i, lv in enumerate(self.levels):
            lv.smoothing = smoothing[min(i, len(smoothing) - 1)]
            lv.cheb = cp.zeros(2 * lv.smoothing - 1, cp.float32)
        self.coarse_inverse = None
        self._floored = None

        # persistent CG state, z is the V-cycle output; scalars hold rz, pAp, rr,
        # rz_old, rz_new
        self.ndof = self.levels[0].ndof
        self.mask = self.levels[0].mask
        self.x, self.r = cp.zeros(self.ndof), cp.zeros(self.ndof)
        self.p, self.Ap = (cp.zeros(self.ndof, cp.float32) for _ in range(2))
        self.scalars = cp.zeros(5)
        self.partial = cp.zeros(DOT_BLOCKS)

    def node_coordinates(self):
        """per axis coordinates of the fine nodes."""
        fine = self.levels[0]
        xi = gauss_lobatto(self.degree)[:-1]
        return [
            np.append((np.arange(e)[:, None] + xi).ravel(), e) * h
            for e, h in zip(fine.elements, fine.h)
        ]

    def face_load(self, axis, side, values):
        """(ndof,) consistent nodal loads of a constant traction/flux on a domain face."""
        fine = self.levels[0]
        xg, wg = gauss(self.degree + 1)
        integrals = wg @ lagrange(gauss_lobatto(self.degree), xg)[0]
        weights = []
        for d in range(self.D):
            w = np.zeros(fine.nodes[d])
            if d == axis:
                w[-1 if side else 0] = 1.0
            else:
                for e in range(fine.elements[d]):
                    w[self.degree * e : self.degree * (e + 1) + 1] += (
                        integrals * fine.h[d]
                    )
            weights.append(w)
        face = weights[0]
        for w in weights[1:]:
            face = np.multiply.outer(face, w)
        return cp.asarray(np.concatenate([v * face.ravel() for v in values]))

    def update(self, coeff):
        """set up operator and hierarchy for a new voxel coefficient grid."""
        cp.cuda.get_current_stream().synchronize()
        with self.stream:
            self._update(coeff)
        self.stream.synchronize()

    def _update(self, coeff):
        coeff = cp.asarray(coeff, dtype=cp.float32)
        e3 = [np.int32(e) for e in list(self.levels[0].elements) + [1] * (3 - self.D)]
        if self.floor and self.sub > 1:
            if self._floored is None:
                self._floored = cp.empty(self.nvoxels, cp.float32)
                self._dfloor = cp.empty(self.nvoxels, cp.float32)
            args = (*e3[1:], np.int32(self.sub), np.float32(self.floor), coeff)
            args += (self._floored, self._dfloor)
            function = kernel(f"element_floor<{self.D}, {FLOOR_POWER}>")
            function((int(np.prod(e3)),), (256,), args)
            coeff = self._floored
        voxel = np.prod([L / n for L, n in zip(self.lengths, self.nvoxels)])
        previous = None
        for lv in self.levels:
            sub = self.nvoxels[0] // lv.elements[0]
            if not lv.fitted:
                W = coeff
            elif sub <= 8:  # from the voxels, no temporaries
                W = lv.weights
                if W is None:
                    W = cp.empty(int(np.prod(lv.elements)) * lv.NQ**self.D, cp.float32)
                L = voxel_table(lv.NQ, sub)
                e = [np.int32(n) for n in list(lv.elements) + [1] * (3 - self.D)]
                args = (*e[1:], np.int32(sub), L, np.float32(voxel * sub**self.D))
                function = kernel(f"fit_voxels<{self.D}, {lv.NQ}>")
                function((int(np.prod(lv.elements)),), (128,), (*args, coeff, W))
            else:  # exactly from the finer level's weights
                R = previous.elements[0] // lv.elements[0]
                W = contract_weights(
                    previous.weights, lv.elements, R, previous.NQ, lv.NQ
                ).ravel()
            if lv.weights is None:
                lv.weights = W.copy() if W is coeff else W
            elif W is not lv.weights:
                lv.weights[...] = W
            previous = lv
        for lv in self.levels[:-1]:
            lv.set_diagonal()
            hi = self.safety * lv.estimate_lmax(self.power_iters)
            cheb = chebyshev_coefficients(hi, self.eig_ratio, lv.smoothing)
            lv.cheb[...] = cp.asarray(cheb, dtype=cp.float32)
        # float32 inverse of the Jacobi-scaled matrix, 10x faster than float64 on
        # consumer GPUs and the same CG iterations down to void 1e-9
        A = self.levels[-1].dense()
        scale = cp.diag(A) ** -0.5
        inverse = cp.linalg.inv((scale[:, None] * A * scale).astype(cp.float32))
        inverse = scale[:, None] * inverse * scale
        if self.coarse_inverse is None:
            self.coarse_inverse = inverse
        else:
            self.coarse_inverse[...] = inverse

    def _by_element(self, field):
        s = self.sub
        return field.reshape([x for n in self.nvoxels for x in (n // s, s)])

    def voxel_energy(self, u):
        """(nvoxels) float32 strain energy of every voxel for unit coefficient."""
        fine, p, s = self.levels[0], self.degree, self.sub
        xg, wg = gauss(p + 1)
        points = np.concatenate([(k + xg) / s for k in range(s)])
        phi, dphi = lagrange(gauss_lobatto(p), points)
        geo = fine.invh + [0.0] * (3 - self.D) + [np.prod(fine.h) / s**self.D]
        tables = cp.asarray(
            np.concatenate([phi.ravel(), dphi.ravel(), wg, geo]), cp.float32
        )
        e3 = [np.int32(e) for e in list(fine.elements) + [1] * (3 - self.D)]
        energy = cp.empty(self.nvoxels, cp.float32)
        name = f"voxel_energy<{self.D}, {self.NF}, {p + 1}, {p + 1}, {s}>"
        args = (*e3, tables, np.float32(self.lam), np.float32(self.mu))
        u = cp.asarray(u, dtype=cp.float64)
        kernel(name)((int(np.prod(fine.elements)),), (128,), (*args, u, energy))
        return energy

    def energy_gradient(self, u):
        """(nvoxels) d(u^T K u) / d(coefficient) for fixed u, through the floor.

        The compliance sensitivity is its negative.
        """
        energy = self.voxel_energy(u)
        if not (self.floor and self.sub > 1):
            return energy
        grouped = self._by_element(energy)
        total = grouped.sum(axis=tuple(range(1, 2 * self.D, 2)), keepdims=True)
        _add_product(self._by_element(self._dfloor), total, grouped)  # no nvoxels temporary
        return energy

    def apply(self, u):
        """float64 A u of the fine operator."""
        out = cp.zeros(self.ndof)
        return self.levels[0].op(cp.asarray(u, dtype=cp.float64), out, T=cp.float64)

    def _smooth(self, lv, pre):
        """Chebyshev on lv.x for rhs lv.b; pre-smoothing starts from x = 0."""
        if pre:
            _cheb_pre(lv.b, lv.dinv, lv.cheb, lv.r, lv.d, lv.x)
        else:
            _cheb_post(lv.b, lv.op(lv.x, lv.Ad), lv.dinv, lv.cheb, lv.r, lv.d, lv.x)
        for k in range(lv.smoothing - 1):
            _cheb_step(
                lv.op(lv.d, lv.Ad), lv.dinv, lv.cheb[2 * k + 1 :], lv.r, lv.d, lv.x
            )
        if pre:
            _subtract(lv.op(lv.d, lv.Ad), lv.r)

    def _transfer(self, name, i, source, target, mask):
        fine, coarse = self.levels[i], self.levels[i + 1]
        R, xi = self.transfers[i]
        f3 = list(fine.nodes) + [1] * (3 - self.D)
        c3 = list(coarse.nodes) + [1] * (3 - self.D)
        args = (*[np.int32(n) for n in f3 + c3], np.int32(R), np.int32(self.NF))
        threads = fine.nn if name == "prolong" else coarse.nn
        launch(kernel(f"{name}<float>"), threads, args + (xi, source, target, mask))

    def _vcycle(self):  # levels[0].b -> levels[0].x
        L = len(self.levels) - 1
        for i in range(L):
            fine, coarse = self.levels[i], self.levels[i + 1]
            self._smooth(fine, pre=True)
            self._transfer("restrict_", i, fine.r, coarse.b, coarse.mask)
        coarse = self.levels[L]
        launch(
            kernel("dense_matvec<float>"),
            coarse.ndof,
            (self.coarse_inverse, coarse.b, coarse.x, np.int32(coarse.ndof)),
        )
        for i in reversed(range(L)):
            fine, coarse = self.levels[i], self.levels[i + 1]
            self._transfer("prolong", i, coarse.x, fine.x, fine.mask)
            self._smooth(fine, pre=False)

    def _dot(self, a, b, out):  # deterministic two-stage reduction
        partial = kernel(f"dot_partial<{ctype(a.dtype)}, {ctype(b.dtype)}>")
        partial((DOT_BLOCKS,), (256,), (a, b, self.partial, np.int64(a.size)))
        final = kernel("dot_final")
        final((1,), (1,), (self.partial, out, np.int32(DOT_BLOCKS)))

    def _precondition(self):  # r -> z = levels[0].x
        _cast(self.r, self.levels[0].b)
        self._vcycle()

    def _iteration(self):  # one flexible PCG step on the persistent state
        x, r, p, Ap, z = self.x, self.r, self.p, self.Ap, self.levels[0].x
        s = self.scalars
        self.levels[0].op(p, Ap)
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
        self.levels[0].op(self.x, r)
        _init_residual(rhs, self.mask, r)
        self._precondition()
        p[...] = z
        self._dot(r, z, self.scalars[0])
        self._dot(r, r, self.scalars[2])

    def solve(self, rhs, x0=None, rtol=1e-8, maxiter=1000):
        """flexible PCG on the fine operator; returns (u, iterations).

        u is the solver's float64 buffer, overwritten by the next solve. No restarts:
        restarting on slow progress discarded the Krylov space and turned slow solves
        on thin-strut designs into 1000 iteration ones.
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
            for it in range(1, maxiter + 1):
                self.graph.launch(self.stream)
                if float(self.scalars[2]) <= tol:
                    break
        self.stream.synchronize()
        return x, it

    def voxel_nodes(self, u):
        """(NF, *voxel nodes) float64 values of u at the voxel corners."""
        fine = self.levels[0]
        s = self.sub
        V, _ = lagrange(gauss_lobatto(self.degree), np.arange(s) / s)
        field = cp.asarray(u, dtype=cp.float64).reshape(self.NF, *fine.nodes)
        for d in range(self.D):
            n, e = fine.nodes[d], fine.elements[d]
            M = np.zeros((e * s + 1, n))
            for k in range(e):
                M[k * s : (k + 1) * s, k * self.degree : (k + 1) * self.degree + 1] = V
            M[-1, -1] = 1.0
            field = cp.moveaxis(
                cp.tensordot(cp.asarray(M), field, axes=([1], [d + 1])), 0, d + 1
            )
        return field
