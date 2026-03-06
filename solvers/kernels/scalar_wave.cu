#include <cuda_fp16.h>

extern "C"
{

// -----------------------------------------------------------------------

__global__ void fd_kernel(const float* __restrict__ u0,
                          const float* __restrict__ u1,
                          float* __restrict__ u2,
                          const float* __restrict__ ind,
                          const float laplace_factor_x,
                          const float laplace_factor_y,
                          const int N, const int M,
                          const int M_padded){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(i > 0 && j > 0 && j < (M - 1) && i < (N - 1))
    {
        const float u1_center = u1[i * M_padded + j]; // load once
        const float ind_center = ind[i * M_padded + j]; // load once

        const float harmonic_x_right = ind[(i + 1) * M_padded + j] /
                                       (ind_center + ind[(i + 1) * M_padded + j]);
        const float harmonic_x_left = ind[(i - 1) * M_padded + j] /
                                       (ind_center + ind[(i - 1) * M_padded + j]);
        const float harmonic_y_right = ind[i * M_padded + (j + 1)] /
                                       (ind_center + ind[i * M_padded + (j + 1)]);
        const float harmonic_y_left = ind[i * M_padded + (j - 1)] /
                                       (ind_center + ind[i * M_padded + (j - 1)]);

        const float laplacian_x = (harmonic_x_right *  (u1[(i + 1) * M_padded + j] - u1_center) -
                                   harmonic_x_left * (u1_center - u1[(i - 1) * M_padded + j]));
        const float laplacian_y = (harmonic_y_right *  (u1[i * M_padded + (j + 1)] - u1_center) -
                                   harmonic_y_left * (u1_center - u1[i * M_padded + (j - 1)]));

        u2[i * M_padded + j] = (-u0[i * M_padded + j] + 2.f * u1_center +
                                laplace_factor_x * laplacian_x +
                                laplace_factor_y * laplacian_y);
    }
}

// -----------------------------------------------------------------------

__global__ void bc_kernel(float *u, const int N, const int M,
                          const int M_padded){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int perimeter = 2 * (N - 2) + 2 * (M - 2);

    if(idx >= perimeter) return;

    // top edge
    if(idx < M - 2)
    {
        const int j = idx + 1;
        u[j] = u[2 * M_padded + j];
    }
    // right edge
    else if(idx < M - 2 + N - 2)
    {
        const int i = idx - (M - 2) + 1;
        u[i * M_padded + (M - 1)] = u[i * M_padded + (M - 3)];
    }
    // bottom edge
    else if(idx < 2 * (M - 2) + N - 2)
    {
        const int j = idx - (M - 2 + N - 2) + 1;
        u[(N - 1) * M_padded + j] = u[(N - 3) * M_padded + j];
    }
    // left edge
    else
    {
        const int i = idx - (2 * (M - 2) + N - 2) + 1;
        u[i * M_padded] = u[i * M_padded + 2];
    }
    // corners are excluded
}

// -----------------------------------------------------------------------

__global__ void excitation_kernel(float *u, const float* source, const int* position,
                                  const int num_source_positions,
                                  const float dt2density, const float* indicator,
                                  const int M_padded){

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_source_positions){
        // TODO strided access could be improved
        const int positioni = position[idx];
        const int positionj = position[idx + num_source_positions];

        u[positioni * M_padded + positionj] += (dt2density /
                                               indicator[positioni * M_padded + positionj] *
                                               source[idx]);
    }
}

// -----------------------------------------------------------------------

__global__ void get_signal_kernel(const float* u, float* um,
                                      const int* sensor_i, const int* sensor_j,
                                      int num_sensors, int M_padded) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sensors) {
        um[idx] = u[sensor_i[idx] * M_padded + sensor_j[idx]];
    }
}

} // extern "C"
