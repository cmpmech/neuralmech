// D2Q9 lattice Boltzmann kernels on a periodic grid, from the first CUDA port to temporal blocking.
// Every kernel stores the deviations g = f - w from rest and relaxes them with two rates. Node
// n = x Ny + y, so y is the fast index. `store` is the storage type (float or __half).
#include <cuda_fp16.h>

#ifndef store
#define store float
#endif

__device__ __forceinline__ int CX(const int i){ const int t[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; return t[i]; }
__device__ __forceinline__ int CY(const int i){ const int t[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1}; return t[i]; }
__device__ __forceinline__ int OPP(const int i){ const int t[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; return t[i]; }
__device__ __forceinline__ int PAIR(const int k){ const int t[4] = {1, 2, 5, 6}; return t[k]; }
__device__ __forceinline__ float W(const int i){ return i == 0 ? 4.f / 9 : (i < 5 ? 1.f / 9 : 1.f / 36); }

__device__ __forceinline__ float load(const float v){ return v; }
__device__ __forceinline__ float load(const __half v){ return __half2float(v); }
__device__ __forceinline__ void save(float &d, const float v){ d = v; }
__device__ __forceinline__ void save(__half &d, const float v){ d = __float2half(v); }

__device__ __forceinline__ int wrap(const int a, const int n){ return a < 0 ? a + n : (a >= n ? a - n : a); }

// two-relaxation-time collision of the deviations g, in place, pair by pair (i, opp(i))
__device__ __forceinline__ void collide(float g[9], const float omega_plus, const float omega_minus){
    float drho = 0, jx = 0, jy = 0;
    #pragma unroll
    for(int i = 0; i < 9; i++){ drho += g[i]; jx += CX(i) * g[i]; jy += CY(i) * g[i]; }
    const float rho = 1 + drho, ux = jx / rho, uy = jy / rho, uu = 1.5f * (ux * ux + uy * uy);

    g[0] -= omega_plus * (g[0] - W(0) * (drho - rho * uu));
    #pragma unroll
    for(int k = 0; k < 4; k++){
        const int i = PAIR(k);
        const float cu = 3 * (CX(i) * ux + CY(i) * uy);
        const float plus = W(i) * (drho + rho * (0.5f * cu * cu - uu)), minus = W(i) * rho * cu;
        const float sym = omega_plus * (0.5f * (g[i] + g[OPP(i)]) - plus);
        const float asym = omega_minus * (0.5f * (g[i] - g[OPP(i)]) - minus);
        g[i] -= sym + asym;
        g[OPP(i)] -= sym - asym;
    }
}

// ------------------------------------ array of structures ------------------------------------
extern "C" __global__ void collide_aos(float *f, const float omega_plus, const float omega_minus,
                                       const int Nx, const int Ny){
    const int x = blockIdx.y, y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= Ny) return;
    const int n = x * Ny + y;

    float g[9];
    for(int i = 0; i < 9; i++) g[i] = f[9 * n + i];
    collide(g, omega_plus, omega_minus);
    for(int i = 0; i < 9; i++) f[9 * n + i] = g[i];
}

extern "C" __global__ void stream_aos(const float *f, float *f_new, const int Nx, const int Ny){
    const int x = blockIdx.y, y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= Ny) return;
    const int n = x * Ny + y;

    for(int i = 0; i < 9; i++){  // push to the periodic neighbor x + c_i
        const int m = wrap(x + CX(i), Nx) * Ny + wrap(y + CY(i), Ny);
        f_new[9 * m + i] = f[9 * n + i];
    }
}

// ------------------------------------ structure of arrays ------------------------------------
extern "C" __global__ void collide_soa(float *f, const float omega_plus, const float omega_minus,
                                       const int Nx, const int Ny){
    const int x = blockIdx.y, y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= Ny) return;
    const int n = x * Ny + y, N = Nx * Ny;

    float g[9];
    for(int i = 0; i < 9; i++) g[i] = f[i * N + n];
    collide(g, omega_plus, omega_minus);
    for(int i = 0; i < 9; i++) f[i * N + n] = g[i];
}

extern "C" __global__ void stream_soa(const float *f, float *f_new, const int Nx, const int Ny){
    const int x = blockIdx.y, y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= Ny) return;
    const int n = x * Ny + y, N = Nx * Ny;

    for(int i = 0; i < 9; i++){
        const int m = wrap(x + CX(i), Nx) * Ny + wrap(y + CY(i), Ny);
        f_new[i * N + m] = f[i * N + n];
    }
}

// ---------------------------------- fused streaming, pull ------------------------------------
extern "C" __global__ void fused_step(const store *__restrict__ f, store *__restrict__ f_new,
                                      const float omega_plus, const float omega_minus,
                                      const int Nx, const int Ny){
    const int x = blockIdx.y, y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= Ny) return;
    const size_t n = (size_t)x * Ny + y, N = (size_t)Nx * Ny;

    float g[9];
    #pragma unroll
    for(int i = 0; i < 9; i++){  // pull from the periodic neighbor x - c_i
        const size_t m = (size_t)wrap(x - CX(i), Nx) * Ny + wrap(y - CY(i), Ny);
        g[i] = load(f[i * N + m]);
    }
    collide(g, omega_plus, omega_minus);
    #pragma unroll
    for(int i = 0; i < 9; i++) save(f_new[i * N + n], g[i]);
}

// ------------------------------------ in place, AA pattern -----------------------------------
// even steps collide in place and store g_i in the slot of the opposite direction
extern "C" __global__ void aa_even(store *f, const float omega_plus, const float omega_minus,
                                   const int Nx, const int Ny){
    const int x = blockIdx.y, y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= Ny) return;
    const size_t n = (size_t)x * Ny + y, N = (size_t)Nx * Ny;

    float g[9];
    #pragma unroll
    for(int i = 0; i < 9; i++) g[i] = load(f[i * N + n]);
    collide(g, omega_plus, omega_minus);
    #pragma unroll
    for(int i = 0; i < 9; i++) save(f[OPP(i) * N + n], g[i]);
}

// odd steps read g_i from the opposite slot of x - c_i and write it to the slot i of x + c_i
extern "C" __global__ void aa_odd(store *f, const float omega_plus, const float omega_minus,
                                  const int Nx, const int Ny){
    const int x = blockIdx.y, y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= Ny) return;
    const size_t N = (size_t)Nx * Ny;

    float g[9];
    #pragma unroll
    for(int i = 0; i < 9; i++){
        const size_t m = (size_t)wrap(x - CX(i), Nx) * Ny + wrap(y - CY(i), Ny);
        g[i] = load(f[OPP(i) * N + m]);
    }
    collide(g, omega_plus, omega_minus);
    #pragma unroll
    for(int i = 0; i < 9; i++){
        const size_t m = (size_t)wrap(x + CX(i), Nx) * Ny + wrap(y + CY(i), Ny);
        save(f[i * N + m], g[i]);
    }
}

// ------------------------------------- temporal blocking -------------------------------------
// each block loads a tile of TILE_X x TILE_Y nodes, advances it by STEPS steps, exchanging the
// moving populations through shared memory, and writes back its inner nodes, STEPS from the edge
#ifndef STEPS
#define STEPS 6
#endif
#ifndef TILE_X
#define TILE_X 48
#endif
#define TILE_Y 64
#define ROWS 8  // threads in x, each owns TILE_X / ROWS nodes
#define NODES (TILE_X / ROWS)

extern "C" __global__ void __launch_bounds__(TILE_Y * ROWS)
temporal_step(const store *__restrict__ f, store *__restrict__ f_new, const float omega_plus,
              const float omega_minus, const int Nx, const int Ny){
    extern __shared__ float shared[];  // a guard row, then the moving populations [8][TILE_X][TILE_Y]
    float *s = shared + TILE_Y + 1;
    const int ty = threadIdx.x, x0 = blockIdx.x * (TILE_X - 2 * STEPS) - STEPS;
    const int y = blockIdx.y * (TILE_Y - 2 * STEPS) - STEPS + ty;
    const size_t N = (size_t)Nx * Ny;

    float g[NODES][9];
    #pragma unroll
    for(int k = 0; k < NODES; k++){
        const size_t n = (size_t)wrap(x0 + threadIdx.y + k * ROWS, Nx) * Ny + wrap(y, Ny);
        #pragma unroll
        for(int i = 0; i < 9; i++) g[k][i] = load(f[i * N + n]);
    }

    for(int t = 0; t < STEPS; t++){
        #pragma unroll
        for(int k = 0; k < NODES; k++){
            const int tx = threadIdx.y + k * ROWS;
            #pragma unroll
            for(int i = 1; i < 9; i++) s[((i - 1) * TILE_X + tx) * TILE_Y + ty] = g[k][i];
        }
        __syncthreads();
        #pragma unroll
        for(int k = 0; k < NODES; k++){
            const int tx = threadIdx.y + k * ROWS;
            #pragma unroll
            for(int i = 1; i < 9; i++)  // nodes at the tile edge pull garbage, which never reaches the inner nodes
                g[k][i] = s[((i - 1) * TILE_X + tx - CX(i)) * TILE_Y + ty - CY(i)];
            collide(g[k], omega_plus, omega_minus);
        }
        __syncthreads();
    }

    if(ty < STEPS || ty >= TILE_Y - STEPS || y >= Ny) return;
    #pragma unroll
    for(int k = 0; k < NODES; k++){
        const int tx = threadIdx.y + k * ROWS, x = x0 + tx;
        if(tx < STEPS || tx >= TILE_X - STEPS || x >= Nx) continue;
        const size_t n = (size_t)x * Ny + y;
        #pragma unroll
        for(int i = 0; i < 9; i++) save(f_new[i * N + n], g[k][i]);
    }
}
