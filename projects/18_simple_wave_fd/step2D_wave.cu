#include <cuda_fp16.h>

extern "C"
{

// -----------------------------------------------------------------------

__global__ void basic_step(const float* __restrict__ u0,
                            const float* __restrict__ u1,
                            float* __restrict__ u2, const float wavespeed,
                            const float dt, const float dx,
                            const float dy, const int N, const int M){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t c = (size_t)i * M + j; // 64-bit once: grids beyond 2^31 points

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        float u1_center = u1[c]; // load once
        float laplacian_x = (u1[c - M] - 2.f * u1_center +
                             u1[c + M]) / (dx * dx);
        float laplacian_y = (u1[c - 1] - 2.f * u1_center +
                             u1[c + 1]) / (dy * dy);
        u2[c] = (-u0[c] + 2.f * u1_center +
                 dt * dt * wavespeed * wavespeed * (laplacian_x +
                                                    laplacian_y));
    }
}

// -----------------------------------------------------------------------

__global__ void coalesced_step(const float* __restrict__ u0,
                            const float* __restrict__ u1,
                            float* __restrict__ u2,
                            const float wavespeed, const float dt,
                            const float dx, const float dy,
                            const int N, const int M){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t c = (size_t)i * M + j; // 64-bit once: grids beyond 2^31 points

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        float u1_center = u1[c]; // load once
        float laplacian_x = (u1[c - M] - 2.f * u1_center +
                             u1[c + M]) / (dx * dx);
        float laplacian_y = (u1[c - 1] - 2.f * u1_center +
                             u1[c + 1]) / (dy * dy);
        u2[c] = (-u0[c] + 2.f * u1_center +
                 dt * dt * wavespeed * wavespeed * (laplacian_x +
                                                    laplacian_y));
    }
}

// -----------------------------------------------------------------------

__global__ void padded_step(const float* __restrict__ u0,
                            const float* __restrict__ u1,
                            float* __restrict__ u2,
                            const float wavespeed, const float dt,
                            const float dx, const float dy,
                            const int N, const int M,
                            const int M_padded){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t c = (size_t)i * M_padded + j; // 64-bit once: grids beyond 2^31 points

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        float u1_center = u1[c]; // load once
        float laplacian_x = (u1[c - M_padded] - 2.f * u1_center +
                             u1[c + M_padded]) / (dx * dx);
        float laplacian_y = (u1[c - 1] - 2.f * u1_center +
                             u1[c + 1]) / (dy * dy);
        u2[c] = (-u0[c] + 2.f * u1_center +
                         dt * dt * wavespeed * wavespeed * (laplacian_x +
                                                            laplacian_y));
    }
}

// -----------------------------------------------------------------------

__global__ void mixed_step(const __half* __restrict__ u0,
                            const __half* __restrict__ u1,
                            __half* __restrict__ u2,
                            const float wavespeed, const float dt,
                            const float dx, const float dy,
                            const int N, const int M,
                            const int M_padded)
{
    // each thread updates the pair of points j and j + 1
    const int j = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t c = (size_t)i * M_padded + j; // 64-bit once: grids beyond 2^31 points

    if(i > 0 && i < (N - 1) && j < M)
    {
        // one 4-byte load fetches both points of a pair
        float2 u0_c = __half22float2(*(const __half2*)&u0[c]);
        float2 u1_c = __half22float2(*(const __half2*)&u1[c]);
        float2 u1_l = __half22float2(*(const __half2*)&u1[c - M_padded]);
        float2 u1_r = __half22float2(*(const __half2*)&u1[c + M_padded]);
        float u1_t = j > 0 ? __half2float(u1[c - 1]) : 0.f;
        float u1_b = __half2float(u1[c + 2]);

        float coefficient = dt * dt * wavespeed * wavespeed;
        float u2_x = -u0_c.x + 2.f * u1_c.x + coefficient *
                     ((u1_l.x - 2.f * u1_c.x + u1_r.x) / (dx * dx) +
                      (u1_t - 2.f * u1_c.x + u1_c.y) / (dy * dy));
        float u2_y = -u0_c.y + 2.f * u1_c.y + coefficient *
                     ((u1_l.y - 2.f * u1_c.y + u1_r.y) / (dx * dx) +
                      (u1_c.x - 2.f * u1_c.y + u1_b) / (dy * dy));

        if(j > 0 && j + 1 < M - 1) // interior pair: one 4-byte store
            *(__half2*)&u2[c] = __floats2half2_rn(u2_x, u2_y);
        else
        {
            if(j > 0 && j < M - 1) u2[c] = __float2half(u2_x);
            if(j + 1 < M - 1) u2[c + 1] = __float2half(u2_y);
        }
    }
}

// -----------------------------------------------------------------------

__global__ void scaled_step(const short* __restrict__ q0,
                            const float* __restrict__ s0,
                            const short* __restrict__ q1,
                            const float* __restrict__ s1,
                            short* __restrict__ q2,
                            float* __restrict__ s2,
                            const float wavespeed, const float dt,
                            const float dx, const float dy,
                            const int N, const int M,
                            const int M_padded)
{
    // each thread updates the pair of points j and j + 1, and the 16 threads
    // of a half-warp share the scale of their block of 32 points
    const int j = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(i <= 0 || i >= N - 1 || j >= M_padded) return; // whole warps: M_padded % 64 == 0

    const size_t c = (size_t)i * M_padded + j; // 64-bit once: grids beyond 2^31 points
    const int S = M_padded / 32; // blocks per row
    const size_t b = (size_t)i * S + j / 32; // block of the pair
    const int lane = threadIdx.x % 32;

    // decode u = s * q, one 4-byte load per pair as in mixed_step
    const short2 p0 = *(const short2*)&q0[c];
    const short2 p1 = *(const short2*)&q1[c];
    const short2 pl = *(const short2*)&q1[c - M_padded];
    const short2 pr = *(const short2*)&q1[c + M_padded];
    float2 u0_c = make_float2(s0[b] * p0.x, s0[b] * p0.y);
    float2 u1_c = make_float2(s1[b] * p1.x, s1[b] * p1.y);
    float2 u1_l = make_float2(s1[b - S] * pl.x, s1[b - S] * pl.y);
    float2 u1_r = make_float2(s1[b + S] * pr.x, s1[b + S] * pr.y);
    // the outer neighbors j - 1 and j + 2 belong to the adjacent threads
    float u1_t = __shfl_up_sync(0xffffffff, u1_c.y, 1);
    float u1_b = __shfl_down_sync(0xffffffff, u1_c.x, 1);
    if(lane == 0) u1_t = j > 0 ? s1[b - 1] * q1[c - 1] : 0.f;
    if(lane == 31) u1_b = j + 2 < M_padded ? s1[b + 1] * q1[c + 2] : 0.f;

    float coefficient = dt * dt * wavespeed * wavespeed;
    float u2_x = -u0_c.x + 2.f * u1_c.x + coefficient *
                 ((u1_l.x - 2.f * u1_c.x + u1_r.x) / (dx * dx) +
                  (u1_t - 2.f * u1_c.x + u1_c.y) / (dy * dy));
    float u2_y = -u0_c.y + 2.f * u1_c.y + coefficient *
                 ((u1_l.y - 2.f * u1_c.y + u1_r.y) / (dx * dx) +
                  (u1_c.x - 2.f * u1_c.y + u1_b) / (dy * dy));
    if(j == 0 || j >= M - 1) u2_x = 0.f; // boundary and padding stay zero
    if(j + 1 >= M - 1) u2_y = 0.f;

    // encode: the largest magnitude of the block sets its scale
    float m = fmaxf(fabsf(u2_x), fabsf(u2_y));
    for(int offset = 1; offset < 16; offset *= 2)
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
    const float inverse = m > 0.f ? 32767.f / m : 0.f;
    *(short2*)&q2[c] = make_short2(__float2int_rn(u2_x * inverse),
                                   __float2int_rn(u2_y * inverse));
    if(lane % 16 == 0) s2[b] = m / 32767.f;
}

// -----------------------------------------------------------------------

__global__ void shared_step(const float* __restrict__ u0,
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

__global__ void temporal_step(const float* __restrict__ u0,
                              const float* __restrict__ u1,
                              float* __restrict__ v0,
                              float* __restrict__ v1,
                              const float wavespeed, const float dt,
                              const float dx, const float dy,
                              const int N, const int M,
                              const int M_padded){
    constexpr int TILE_J = 128; // = blockDim.x
    constexpr int TILE_I = 48;
    constexpr int STEPS = 8; // time steps per launch and halo width, even

    // the tile starts STEPS points before the points the block updates
    const int tj = threadIdx.x;
    const int j = blockIdx.x * (TILE_J - 2 * STEPS) + tj - STEPS;
    const int i0 = blockIdx.y * (TILE_I - 2 * STEPS) - STEPS;

    __shared__ float u_s[2][TILE_I][TILE_J]; // both time levels
    for(int ti = threadIdx.y; ti < TILE_I; ti += blockDim.y){
        const int i = i0 + ti;
        const bool inside = i >= 0 && i < N && j >= 0 && j < M;
        const size_t c = (size_t)i * M_padded + j;
        u_s[0][ti][tj] = inside ? u0[c] : 0.f;
        u_s[1][ti][tj] = inside ? u1[c] : 0.f;
    }
    __syncthreads();

    for(int s = 0; s < STEPS; s++){
        float (*u0_s)[TILE_J] = u_s[s % 2]; // overwritten by u2
        float (*u1_s)[TILE_J] = u_s[1 - s % 2];
        for(int ti = threadIdx.y; ti < TILE_I; ti += blockDim.y){
            const int i = i0 + ti;
            // the valid points shrink by one per step from the tile edges
            if(ti > s && ti < TILE_I - 1 - s && tj > s && tj < TILE_J - 1 - s &&
               i > 0 && j > 0 && i < N - 1 && j < M - 1)
            {
                float u1_center = u1_s[ti][tj]; // load once
                float laplacian_x = (u1_s[ti - 1][tj] - 2.f * u1_center +
                                     u1_s[ti + 1][tj]) / (dx * dx);
                float laplacian_y = (u1_s[ti][tj - 1] - 2.f * u1_center +
                                     u1_s[ti][tj + 1]) / (dy * dy);
                u0_s[ti][tj] = (-u0_s[ti][tj] + 2.f * u1_center +
                                 dt * dt * wavespeed * wavespeed * (laplacian_x +
                                                                    laplacian_y));
            }
        }
        __syncthreads();
    }

    // after an even number of steps, u_s[0] holds the older and u_s[1] the newer level
    if(tj >= STEPS && tj < TILE_J - STEPS && j < M){
        for(int ti = STEPS + threadIdx.y; ti < TILE_I - STEPS; ti += blockDim.y){
            const int i = i0 + ti;
            if(i < N){
                const size_t c = (size_t)i * M_padded + j;
                v0[c] = u_s[0][ti][tj];
                v1[c] = u_s[1][ti][tj];
            }
        }
    }
}

} // extern "C"
