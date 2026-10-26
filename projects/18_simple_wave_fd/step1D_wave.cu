extern "C"
{

__global__ void step1D(const float* __restrict__ u0,
                       const float* __restrict__ u1,
                       float* __restrict__ u2, const float wavespeed,
                       const float dt, const float dx, const int N){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i > 0 && i < (N - 1))
    {
        float u1_center = u1[i]; // load once
        float laplacian_x = (u1[i - 1] - 2.f * u1_center + u1[i + 1]) / (dx * dx);
        u2[i] = (-u0[i] + 2.f * u1_center +
                 dt * dt * wavespeed * wavespeed * laplacian_x);
    }
}

} // extern "C"
