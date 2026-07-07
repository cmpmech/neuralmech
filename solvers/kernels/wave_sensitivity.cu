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
