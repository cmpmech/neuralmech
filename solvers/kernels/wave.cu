// Generalized scalar/acoustic/elastic wave finite-difference kernels.
//
// Compile-time configuration (set via -D flags from wave.py):
//   USE_FLOAT             single precision (real_t = float); default is double.
//   NDIM = 1 | 2 | 3      spatial dimension.
//   FORMULATION_ACOUSTIC  acoustic wave equation.
//   FORMULATION_ELASTIC   isotropic elastic wave equation (NDIM-component vector
//                         displacement, NDIM in {2, 3}); default is the scalar eq.

#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

#ifndef FORMULATION_ELASTIC
__device__ __forceinline__ real_t flux_axis(const real_t *__restrict__ u1,
                                            const real_t *__restrict__ mat,
                                            const int idx, const int s,
                                            const real_t uc, const real_t matc,
                                            const real_t factor) {
  const real_t up = u1[idx + s];
  const real_t um = u1[idx - s];
  const real_t mp = mat[idx + s];
  const real_t mm = mat[idx - s];
#ifdef FORMULATION_ACOUSTIC
  const real_t gp = 1.f / (matc + mp);
  const real_t gm = 1.f / (matc + mm);
#else
  const real_t gp = mp / (matc + mp);
  const real_t gm = mm / (matc + mm);
#endif
  return factor * (gp * (up - uc) - gm * (uc - um));
}
#endif

#ifdef FORMULATION_ELASTIC
// Isotropic elastic stress divergence via conservative half-node fluxes. The
// displacement is stored as NDIM contiguous component blocks of comp_stride
// (cs) each, so component c at grid offset "off" is u1[c * cs + idx + off].
// gamma is a per-node presence field 0 < gamma <= 1 that scales BOTH inertia
// (gamma * rho) and stiffness (gamma * C), so a uniform gamma cancels and only
// its gradients scatter -- mirroring the scalar formulation.

__device__ __forceinline__ real_t harmonic(const real_t a, const real_t b) {
  return 2.f * a * b / (a + b);
}

// stress vector sigma_{i,j} (i = 0..NDIM-1, fixed normal axis j) on the cell face
// between the node and its neighbour at offset nn along axis j. sgn is +1 for the
// upper face (nn = +o[j]) and -1 for the lower face. Diagonal strains use the
// compact normal difference; off-diagonal strains average the node and neighbour
// central differences. gface returns the harmonic-averaged gamma on the face.
__device__ void face_stress(const real_t *__restrict__ u1, const int idx,
                            const int cs, const real_t *__restrict__ gamma,
                            const int *o, const real_t *invdx, const int j,
                            const int nn, const int sgn, const real_t lam,
                            const real_t mu, real_t *sig, real_t *gface) {
  real_t d[NDIM][NDIM]; // d[c][a] = d u_c / d x_a on the face
#pragma unroll
  for (int c = 0; c < NDIM; ++c)
#pragma unroll
    for (int a = 0; a < NDIM; ++a) {
      if (a == j)
        d[c][a] = sgn * (u1[c * cs + idx + nn] - u1[c * cs + idx]) * invdx[a];
      else
        d[c][a] = 0.25f * invdx[a] *
                  ((u1[c * cs + idx + o[a]] - u1[c * cs + idx - o[a]]) +
                   (u1[c * cs + idx + nn + o[a]] - u1[c * cs + idx + nn - o[a]]));
    }
  real_t theta = 0.f;
#pragma unroll
  for (int k = 0; k < NDIM; ++k)
    theta += d[k][k];
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
    sig[i] = lam * (i == j ? theta : 0.f) + mu * (d[i][j] + d[j][i]);
  *gface = harmonic(gamma[idx], gamma[idx + nn]);
}
#endif

extern "C" {

// ------------------------------------------------------------------------------------

#ifndef FORMULATION_ELASTIC
__global__ void fd_kernel(const real_t *__restrict__ u0,
                          const real_t *__restrict__ u1,
                          real_t *__restrict__ u2,
#ifdef FORMULATION_ACOUSTIC
                          const real_t *__restrict__ rho,
                          const real_t *__restrict__ kappa,
                          const real_t *__restrict__ damping, const real_t dt,
#else
                          const real_t *__restrict__ gamma,
#endif
                          const real_t f0, const int N0
#if NDIM >= 2
                          ,
                          const real_t f1, const int N1, const int s0
#endif
#if NDIM >= 3
                          ,
                          const real_t f2, const int N2, const int s1
#endif
) {
#if NDIM == 1
  const int a0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(a0 > 0 && a0 < N0 - 1))
    return;
  const int idx = a0;
#elif NDIM == 2
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1))
    return;
  const int idx = a0 * s0 + a1;
#elif NDIM == 3
  const int a2 = blockIdx.x * blockDim.x + threadIdx.x;
  const int a1 = blockIdx.y * blockDim.y + threadIdx.y;
  const int a0 = blockIdx.z * blockDim.z + threadIdx.z;
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&
        a2 < N2 - 1))
    return;
  const int idx = a0 * s0 + a1 * s1 + a2;
#endif

  const real_t uc = u1[idx]; // load once

#ifdef FORMULATION_ACOUSTIC
  const real_t matc = rho[idx];
#else
  const real_t matc = gamma[idx];
  const real_t *rho = gamma; // unify the flux_axis material argument below
#endif

#if NDIM == 1
  real_t laplacian = flux_axis(u1, rho, idx, 1, uc, matc, f0);
#elif NDIM == 2
  real_t laplacian = flux_axis(u1, rho, idx, s0, uc, matc, f0) +
                     flux_axis(u1, rho, idx, 1, uc, matc, f1);
#elif NDIM == 3
  real_t laplacian = flux_axis(u1, rho, idx, s0, uc, matc, f0) +
                     flux_axis(u1, rho, idx, s1, uc, matc, f1) +
                     flux_axis(u1, rho, idx, 1, uc, matc, f2);
#endif

#ifdef FORMULATION_ACOUSTIC
  const real_t kc = kappa[idx];
  const real_t beta = 0.5f * kc * damping[idx] * dt;
  u2[idx] = (2.f * uc - u0[idx] * (1.f - beta) + kc * laplacian) / (1.f + beta);
#else
  u2[idx] = -u0[idx] + 2.f * uc + laplacian;
#endif
}
#else // FORMULATION_ELASTIC

__global__ void fd_kernel(const real_t *__restrict__ u0,
                          const real_t *__restrict__ u1,
                          real_t *__restrict__ u2,
                          const real_t *__restrict__ gamma, const int cs,
                          const real_t lam, const real_t mu, const real_t rho,
                          const real_t dt2, const real_t f0, const int N0
#if NDIM >= 2
                          ,
                          const real_t f1, const int N1, const int s0
#endif
#if NDIM >= 3
                          ,
                          const real_t f2, const int N2, const int s1
#endif
) {
#if NDIM == 2
  const int a1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int a0 = blockIdx.y * blockDim.y + threadIdx.y;
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1))
    return;
  const int idx = a0 * s0 + a1;
  const int o[2] = {s0, 1};
  const real_t invdx[2] = {f0, f1};
#elif NDIM == 3
  const int a2 = blockIdx.x * blockDim.x + threadIdx.x;
  const int a1 = blockIdx.y * blockDim.y + threadIdx.y;
  const int a0 = blockIdx.z * blockDim.z + threadIdx.z;
  if (!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 &&
        a2 < N2 - 1))
    return;
  const int idx = a0 * s0 + a1 * s1 + a2;
  const int o[3] = {s0, s1, 1};
  const real_t invdx[3] = {f0, f1, f2};
#endif

  real_t div[NDIM];
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
    div[i] = 0.f;

#pragma unroll
  for (int j = 0; j < NDIM; ++j) {
    real_t sp[NDIM], sm[NDIM], gp, gm;
    face_stress(u1, idx, cs, gamma, o, invdx, j, o[j], 1, lam, mu, sp, &gp);
    face_stress(u1, idx, cs, gamma, o, invdx, j, -o[j], -1, lam, mu, sm, &gm);
#pragma unroll
    for (int i = 0; i < NDIM; ++i)
      div[i] += (gp * sp[i] - gm * sm[i]) * invdx[j];
  }

  const real_t scale = dt2 / (gamma[idx] * rho);
#pragma unroll
  for (int i = 0; i < NDIM; ++i)
    u2[i * cs + idx] = 2.f * u1[i * cs + idx] - u0[i * cs + idx] + scale * div[i];
}
#endif // FORMULATION_ELASTIC

// ------------------------------------------------------------------------------------
// homogeneous Neumann: mirror each ghost face onto the interior cell two in.
// one launch per axis; the grid covers the interior range of the other axes
// (corners are left untouched)

__global__ void bc_kernel(real_t *__restrict__ u, const int axis, const int N0
#if NDIM >= 2
                          ,
                          const int N1, const int s0
#endif
#if NDIM >= 3
                          ,
                          const int N2, const int s1
#endif
) {
#if NDIM == 1
  u[0] = u[2];
  u[N0 - 1] = u[N0 - 3];
#elif NDIM == 2
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (axis == 0) {
    const int a1 = t + 1;
    if (a1 > N1 - 2)
      return;
    u[a1] = u[2 * s0 + a1];
    u[(N0 - 1) * s0 + a1] = u[(N0 - 3) * s0 + a1];
  } else {
    const int a0 = t + 1;
    if (a0 > N0 - 2)
      return;
    u[a0 * s0] = u[a0 * s0 + 2];
    u[a0 * s0 + (N1 - 1)] = u[a0 * s0 + (N1 - 3)];
  }
#elif NDIM == 3
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int q = blockIdx.y * blockDim.y + threadIdx.y;
  if (axis == 0) {
    const int a1 = q + 1, a2 = p + 1;
    if (a1 > N1 - 2 || a2 > N2 - 2)
      return;
    const int off = a1 * s1 + a2;
    u[off] = u[2 * s0 + off];
    u[(N0 - 1) * s0 + off] = u[(N0 - 3) * s0 + off];
  } else if (axis == 1) {
    const int a0 = q + 1, a2 = p + 1;
    if (a0 > N0 - 2 || a2 > N2 - 2)
      return;
    const int off = a0 * s0 + a2;
    u[off] = u[off + 2 * s1];
    u[off + (N1 - 1) * s1] = u[off + (N1 - 3) * s1];
  } else {
    const int a0 = q + 1, a1 = p + 1;
    if (a0 > N0 - 2 || a1 > N1 - 2)
      return;
    const int off = a0 * s0 + a1 * s1;
    u[off] = u[off + 2];
    u[off + (N2 - 1)] = u[off + (N2 - 3)];
  }
#endif
}

// ------------------------------------------------------------------------------------

__global__ void excitation_kernel(real_t *__restrict__ u,
                                  const real_t *__restrict__ source,
                                  const int *__restrict__ lin_index,
                                  const int num_sources, const real_t dt2,
                                  const real_t *__restrict__ scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sources) {
    const int li = lin_index[idx];
#ifdef FORMULATION_ACOUSTIC
    u[li] += scale[li] * dt2 * source[idx]; // scale = kappa
#else
    u[li] += dt2 / scale[li] * source[idx]; // scale = gamma, dt2 = dt^2/rho0
#endif
  }
}

// ------------------------------------------------------------------------------------

__global__ void get_signal_kernel(const real_t *__restrict__ u,
                                  real_t *__restrict__ um,
                                  const int *__restrict__ lin_index,
                                  const int num_sensors) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_sensors) {
    um[idx] = u[lin_index[idx]];
  }
}

} // extern "C"
