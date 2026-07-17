// Adjoint sensitivity kernels for the scalar / acoustic wave equation (see wave.cu
// for the forward kernels and the compile-time configuration flags). Generalized
// to NDIM dimensions. One integrand kernel serves both sensitivity variants: the
// Frechet kernel is the bilinear form (Eq. 10 scalar / Eq. 20 acoustic), evaluated
// on a forward field u and an adjoint field lambda, with formulation-specific
// coefficients passed in:
//   scale ( coef_t (du/dt)(dlambda/dt) + coef_grad (grad u . grad lambda) )
// The lossless superposition variant (wave_sensitivity.py) is the diagonal case:
// it aliases lambda = u (same device pointers for both triplets), recovering the
// quadratic form coef_t (du/dt)^2 + coef_grad |grad u|^2, and uses scale = +/-1 to
// add/subtract contributions. The boundary-reconstruction variant passes distinct
// fields and scale = 1.

#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

extern "C"
{

// -----------------------------------------------------------------------

#ifdef FORMULATION_ELASTIC
// Elastic Frechet kernel: the design field gamma scales inertia and stiffness
// alike (see wave.cu), so its sensitivity density is the kinetic minus strain
// bilinear    coef_t (du/dt).(dlambda/dt) + eps(u):C:eps(lambda)
// with coef_t = -rho, C the isotropic tensor (lam, mu), and the displacement
// stored as NDIM component blocks of comp_stride (cs). Central differences at
// the node; l aliased to u recovers the quadratic form for the superposition
// variant.
__global__ void integrand_step_kernel(real_t* __restrict__ kernel,
                                      const real_t* __restrict__ u0,
                                      const real_t* __restrict__ u1,
                                      const real_t* __restrict__ u2,
                                      const real_t* __restrict__ l0,
                                      const real_t* __restrict__ l1,
                                      const real_t* __restrict__ l2,
                                      const real_t dt, const real_t coef_t,
                                      const real_t lam, const real_t mu,
                                      const int cs, const real_t scale,
                                      const real_t dx0, const int N0
#if NDIM >= 2
                                      , const real_t dx1, const int N1, const int s0
#endif
#if NDIM >= 3
                                      , const real_t dx2, const int N2, const int s1
#endif
                                      ){
#if NDIM == 2
    const int a1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int a0 = blockIdx.y * blockDim.y + threadIdx.y;
    if(!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1)) return;
    const int idx = a0 * s0 + a1;
    const int o[2] = {s0, 1};
    const real_t hidx[2] = {0.5f / dx0, 0.5f / dx1};
#elif NDIM == 3
    const int a2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int a1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int a0 = blockIdx.z * blockDim.z + threadIdx.z;
    if(!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 && a2 < N2 - 1)) return;
    const int idx = a0 * s0 + a1 * s1 + a2;
    const int o[3] = {s0, s1, 1};
    const real_t hidx[3] = {0.5f / dx0, 0.5f / dx1, 0.5f / dx2};
#endif

    real_t gu[NDIM][NDIM], gl[NDIM][NDIM];
#pragma unroll
    for(int c = 0; c < NDIM; ++c)
#pragma unroll
        for(int a = 0; a < NDIM; ++a){
            gu[c][a] = (u1[c * cs + idx + o[a]] - u1[c * cs + idx - o[a]]) * hidx[a];
            gl[c][a] = (l1[c * cs + idx + o[a]] - l1[c * cs + idx - o[a]]) * hidx[a];
        }
    real_t thu = 0.f, thl = 0.f;
#pragma unroll
    for(int k = 0; k < NDIM; ++k){ thu += gu[k][k]; thl += gl[k][k]; }
    real_t ee = 0.f;
#pragma unroll
    for(int i = 0; i < NDIM; ++i)
#pragma unroll
        for(int j = 0; j < NDIM; ++j)
            ee += 0.25f * (gu[i][j] + gu[j][i]) * (gl[i][j] + gl[j][i]);
    real_t kin = 0.f;
#pragma unroll
    for(int i = 0; i < NDIM; ++i)
        kin += (u2[i * cs + idx] - u0[i * cs + idx]) * (l2[i * cs + idx] - l0[i * cs + idx]);
    kin *= coef_t / (4.f * dt * dt);

    kernel[idx] += scale * (kin + lam * thu * thl + 2.f * mu * ee);
}
#else

__global__ void integrand_step_kernel(real_t* __restrict__ kernel,
                                      const real_t* __restrict__ u0,
                                      const real_t* __restrict__ u1,
                                      const real_t* __restrict__ u2,
                                      const real_t* __restrict__ l0,
                                      const real_t* __restrict__ l1,
                                      const real_t* __restrict__ l2,
                                      const real_t dt, const real_t coef_t,
                                      const real_t coef_grad, const real_t scale,
                                      const real_t dx0, const int N0
#if NDIM >= 2
                                      , const real_t dx1, const int N1, const int s0
#endif
#if NDIM >= 3
                                      , const real_t dx2, const int N2, const int s1
#endif
                                      ){
#if NDIM == 1
    const int a0 = blockIdx.x * blockDim.x + threadIdx.x;
    if(!(a0 > 0 && a0 < N0 - 1)) return;
    const int idx = a0;
    const int o0 = 1;
#elif NDIM == 2
    const int a1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int a0 = blockIdx.y * blockDim.y + threadIdx.y;
    if(!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1)) return;
    const int idx = a0 * s0 + a1;
    const int o0 = s0, o1 = 1;
#elif NDIM == 3
    const int a2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int a1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int a0 = blockIdx.z * blockDim.z + threadIdx.z;
    if(!(a0 > 0 && a0 < N0 - 1 && a1 > 0 && a1 < N1 - 1 && a2 > 0 && a2 < N2 - 1)) return;
    const int idx = a0 * s0 + a1 * s1 + a2;
    const int o0 = s0, o1 = s1, o2 = 1;
#endif

    const real_t dudt = (u2[idx] - u0[idx]) / (2.f * dt);
    const real_t dldt = (l2[idx] - l0[idx]) / (2.f * dt);

    real_t graddot = 0.f;
    const real_t gu0 = (u1[idx + o0] - u1[idx - o0]) / (2.f * dx0);
    const real_t gl0 = (l1[idx + o0] - l1[idx - o0]) / (2.f * dx0);
    graddot += gu0 * gl0;
#if NDIM >= 2
    const real_t gu1 = (u1[idx + o1] - u1[idx - o1]) / (2.f * dx1);
    const real_t gl1 = (l1[idx + o1] - l1[idx - o1]) / (2.f * dx1);
    graddot += gu1 * gl1;
#endif
#if NDIM >= 3
    const real_t gu2 = (u1[idx + o2] - u1[idx - o2]) / (2.f * dx2);
    const real_t gl2 = (l1[idx + o2] - l1[idx - o2]) / (2.f * dx2);
    graddot += gu2 * gl2;
#endif

    kernel[idx] += scale * (coef_t * dudt * dldt + coef_grad * graddot);
}
#endif // FORMULATION_ELASTIC

// -----------------------------------------------------------------------
// adjoint source = -(measured - simulated) at each sensor; sensors are addressed
// through precomputed linear indices, so this kernel is dimension-agnostic.

__global__ void adjoint_source_step_kernel(real_t* __restrict__ fadjoint,
                                          real_t* __restrict__ fadjoint_squared,
                                          const real_t* __restrict__ u,
                                          const real_t* __restrict__ um,
                                          const int* __restrict__ lin_index,
                                          const int num_sensors){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_sensors) return;

    const real_t us = u[lin_index[idx]];
    fadjoint[idx] = -(us - um[idx]);
    fadjoint_squared[idx] += fadjoint[idx] * fadjoint[idx];
}

} // extern "C"
