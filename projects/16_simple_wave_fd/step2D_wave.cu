#include <cuda_fp16.h>

extern "C"
{

// -----------------------------------------------------------------------

__global__ void fd_kernelV1(const float* __restrict__ u0,
                            const float* __restrict__ u1,
                            float* __restrict__ u2, const float wavespeed,
                            const float dt, const float dx,
                            const float dy, const int N, const int M){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        float u1_center = u1[i * M + j]; // load once
        float laplacian_x = (u1[(i - 1) * M + j] - 2.f * u1_center +
                             u1[(i + 1) * M + j]) / (dx * dx);
        float laplacian_y = (u1[i * M + (j - 1)] - 2.f * u1_center +
                             u1[i * M + (j + 1)]) / (dy * dy);
        u2[i * M + j] = (-u0[i * M + j] + 2.f * u1_center +
                         dt * dt * wavespeed * wavespeed * (laplacian_x +
                                                            laplacian_y));
    }
}

// -----------------------------------------------------------------------

__global__ void fd_kernelV2(const float* __restrict__ u0,
                            const float* __restrict__ u1,
                            float* __restrict__ u2,
                            const float wavespeed, const float dt,
                            const float dx, const float dy,
                            const int N, const int M){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        float u1_center = u1[i * M + j]; // load once
        float laplacian_x = (u1[(i - 1) * M + j] - 2.f * u1_center +
                             u1[(i + 1) * M + j]) / (dx * dx);
        float laplacian_y = (u1[i * M + (j - 1)] - 2.f * u1_center +
                             u1[i * M + (j + 1)]) / (dy * dy);
        u2[i * M + j] = (-u0[i * M + j] + 2.f * u1_center +
                         dt * dt * wavespeed * wavespeed * (laplacian_x +
                                                            laplacian_y));
    }
}

// -----------------------------------------------------------------------

__global__ void fd_kernelV3(const float* __restrict__ u0,
                            const float* __restrict__ u1,
                            float* __restrict__ u2,
                            const float wavespeed, const float dt,
                            const float dx, const float dy,
                            const int N, const int M,
                            const int M_padded){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        float u1_center = u1[i * M_padded + j]; // load once
        float laplacian_x = (u1[(i - 1) * M_padded + j] - 2.f * u1_center +
                             u1[(i + 1) * M_padded + j]) / (dx * dx);
        float laplacian_y = (u1[i * M_padded + (j - 1)] - 2.f * u1_center +
                             u1[i * M_padded + (j + 1)]) / (dy * dy);
        u2[i * M_padded + j] = (-u0[i * M_padded + j] + 2.f * u1_center +
                         dt * dt * wavespeed * wavespeed * (laplacian_x +
                                                            laplacian_y));
    }
}

// -----------------------------------------------------------------------

__global__ void fd_kernelV4(const float* __restrict__ u0,
                            const float* __restrict__ u1,
                            float* __restrict__ u2,
                            const float wavespeed, const float dt,
                            const float dx, const float dy,
                            const int N, const int M, const int M_padded){
    constexpr int BLOCK_DIM_X = 128;
    constexpr int BLOCK_DIM_Y = 4;
    constexpr int HALO = 1;

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    int tj = threadIdx.x;
    int ti = threadIdx.y;

    __shared__ float u1_s[BLOCK_DIM_Y + HALO * 2][BLOCK_DIM_X + HALO * 2];
    // load interior data
    if(i < N && j < M){
        u1_s[ti + HALO][tj + HALO] = u1[i * M_padded + j];
    }
    // load left HALO (first two x threads)
    if(tj < HALO && j >= HALO){
        u1_s[ti + HALO][tj] = u1[i * M_padded + (j - HALO)];
    }
    // load right HALO (last two x threads)
    if(tj >= BLOCK_DIM_X - HALO && j + HALO < M){
        u1_s[ti + HALO][tj + 2 * HALO] = u1[i * M_padded + (j + HALO)];
    }
    // load top HALO (first two y threads)
    if(ti < HALO && i >= HALO){
        u1_s[ti][tj + HALO] = u1[(i - HALO) * M_padded + j];
    }
    // load bot HALO (last two y threads)
    if(ti >= BLOCK_DIM_Y - HALO && i + HALO < N){
        u1_s[ti + 2 * HALO][tj + HALO] = u1[(i + HALO) * M_padded + j];
    }
    // corners are not needed

    __syncthreads();

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        int sj = tj + HALO; // shifted index
        int si = ti + HALO; // shifted index
        float u1_center = u1_s[si][sj]; // load once
        float laplacian_x = (u1_s[si - 1][sj] - 2.f * u1_center +
                             u1_s[si + 1][sj]) / (dx * dx);
        float laplacian_y = (u1_s[si][sj - 1] - 2.f * u1_center +
                             u1_s[si][sj + 1]) / (dy * dy);
        u2[i * M_padded + j] = (-u0[i * M_padded + j] + 2.f * u1_center +
                         dt * dt * wavespeed * wavespeed * (laplacian_x +
                                                            laplacian_y));
    }
}

// -----------------------------------------------------------------------

__global__ void fd_kernelV5(const __half* __restrict__ u0,
                            const __half* __restrict__ u1,
                            __half* __restrict__ u2,
                            const float wavespeed, const float dt,
                            const float dx, const float dy,
                            const int N, const int M,
                            const int M_padded)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
//     const int M_padded = blockDim.x * gridDim.x;

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        float u0_c = __half2float(u0[i * M_padded + j]);
        float u1_c = __half2float(u1[i * M_padded + j]);
        float u1_l = __half2float(u1[(i - 1) * M_padded + j]);
        float u1_r = __half2float(u1[(i + 1) * M_padded + j]);
        float u1_t = __half2float(u1[i * M_padded + (j - 1)]);
        float u1_b = __half2float(u1[i * M_padded + (j + 1)]);

        float laplacian_x = (u1_l - 2.f * u1_c +
                             u1_r) / (dx * dx);
        float laplacian_y = (u1_t - 2.f * u1_c +
                             u1_b) / (dy * dy);
        float u2_c = (-u0_c + 2.f * u1_c +
                      dt * dt * wavespeed * wavespeed *
                      (laplacian_x + laplacian_y));
        u2[i * M_padded + j] = __float2half(u2_c);
    }
}

} // extern "C"
