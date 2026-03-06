extern "C"
{

__global__ void integrand_step_kernel(float* kernel, const float* u0, const float* u1,
                                      const float* u2, const float dt, const float dx,
                                      const float dy, const float density, const float densitywavespeed2,
                                      const float sign, const int N, const int M, const int M_padded)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < (N - 1) && j > 0 && j < (M - 1)){

    const float dudt = (u2[i * M_padded + j] - u0[i * M_padded + j]) / 2.f / dt;
    const float dudx = (u1[(i + 1) * M_padded + j] - u1[(i - 1) * M_padded + j]) / 2.f / dx;
    const float dudy = (u1[i * M_padded + (j + 1)] - u1[i * M_padded + (j - 1)]) / 2.f / dy;

    kernel[i * M_padded + j] +=  sign * (-density * dudt * dudt +
                                         densitywavespeed2 * (dudx * dudx + dudy * dudy));
    }
}

__global__ void adjoint_source_step_kernel(float* fadjoint, float* fadjoint_squared,
                                           const float* u, const float* um, const int* sensors,
                                           const int num_sensors, const int M_padded)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= num_sensors)
    {
        return;
    }
    // TODO strided access could be improved
    const int sensor_posi = sensors[idx];
    const int sensor_posj = sensors[idx + num_sensors];

    const float us = u[sensor_posi * M_padded + sensor_posj];
    fadjoint[idx] = - (us - um[idx]); // only temporal slice is passed of both um and fadjoint

    fadjoint_squared[idx] += fadjoint[idx] * fadjoint[idx];

}

} // extern "C"