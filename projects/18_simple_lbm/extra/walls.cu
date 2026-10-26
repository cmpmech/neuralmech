// Appended to ../lbm.cu: temporal blocking with bounce-back from solid nodes, and a sponge that
// relaxes the outflow columns toward the inflow, so that the periodic grid acts as a channel.

// wall: bit i set where the node x - c_i is solid, so bit 0 marks a solid node itself
extern "C" __global__ void __launch_bounds__(TILE_Y * ROWS)
temporal_walls_step(const store *__restrict__ f, store *__restrict__ f_new,
                    const unsigned short *__restrict__ wall, const float omega_plus,
                    const float omega_minus, const int Nx, const int Ny){
    extern __shared__ float shared[];
    float *s = shared + TILE_Y + 1;
    const int ty = threadIdx.x, x0 = blockIdx.x * (TILE_X - 2 * STEPS) - STEPS;
    const int y = blockIdx.y * (TILE_Y - 2 * STEPS) - STEPS + ty;
    const size_t N = (size_t)Nx * Ny;

    float g[NODES][9];
    unsigned short mask[NODES];
    #pragma unroll
    for(int k = 0; k < NODES; k++){
        const size_t n = (size_t)wrap(x0 + threadIdx.y + k * ROWS, Nx) * Ny + wrap(y, Ny);
        mask[k] = wall[n];
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
            float p[9];
            p[0] = g[k][0];
            #pragma unroll
            for(int i = 1; i < 9; i++){  // bounce-back takes the opposite population of the node itself
                const float pulled = s[((i - 1) * TILE_X + tx - CX(i)) * TILE_Y + ty - CY(i)];
                p[i] = mask[k] >> i & 1 ? g[k][OPP(i)] : pulled;
            }
            if(!(mask[k] & 1)){
                collide(p, omega_plus, omega_minus);
                #pragma unroll
                for(int i = 0; i < 9; i++) g[k][i] = p[i];
            }
        }
        __syncthreads();
    }

    if(ty < STEPS || ty >= TILE_Y - STEPS || y >= Ny) return;
    #pragma unroll
    for(int k = 0; k < NODES; k++){
        const int tx = threadIdx.y + k * ROWS, x = x0 + tx;
        if(tx < STEPS || tx >= TILE_X - STEPS || x >= Nx || mask[k] & 1) continue;
        const size_t n = (size_t)x * Ny + y;
        #pragma unroll
        for(int i = 0; i < 9; i++) save(f_new[i * N + n], g[k][i]);
    }
}

// relaxes the columns x >= start toward the equilibrium at the inflow velocity u, by sigma at most
extern "C" __global__ void sponge(store *f, const float u, const float sigma, const int start,
                                  const int Nx, const int Ny){
    const int x = start + blockIdx.y, y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= Ny || x >= Nx) return;
    const size_t n = (size_t)x * Ny + y, N = (size_t)Nx * Ny;
    const float ramp = sinpif(0.5f * (x - start) / (Nx - start)), r = sigma * ramp * ramp;

    #pragma unroll
    for(int i = 0; i < 9; i++){
        const float cu = 3 * CX(i) * u, eq = W(i) * (cu + 0.5f * cu * cu - 1.5f * u * u), g = load(f[i * N + n]);
        save(f[i * N + n], g + r * (eq - g));
    }
}
