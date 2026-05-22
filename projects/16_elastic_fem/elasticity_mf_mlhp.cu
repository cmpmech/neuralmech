// Matrix-free K*U kernels for structured voxel FEM built on mlhp.
//
// Differences from elasticity_mf.cu:
//   - EFT passed as an explicit device array (supports any polynomial degree)
//   - K_ref passed as a device pointer (not constant memory), any ndof_e
//   - Double precision throughout
//   - No hardcoded Q1 geometry: both 2D and 3D handled identically
//
// Requires compute capability >= 6.0 for native double atomicAdd.

extern "C"
{

#define MAX_NDOF_E 64   // enough for p=3 in 3D (48 DOFs per element)

// double atomicAdd emulation for cc < 6.0
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* addr_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// One thread per element.
// u   [ndof]           input DOF vector
// Ku  [ndof]           output, must be zeroed before launch
// E_values [n_elem]    per-element Young's modulus
// efts     [n_elem * ndof_e]  element freedom tables (row-major)
// K_ref    [ndof_e * ndof_e]  reference element stiffness with E=1 (row-major)
__global__ void cuda_matvec(const double* __restrict__ u,
                            double*       Ku,
                            const double* __restrict__ E_values,
                            const int*    __restrict__ efts,
                            const double* __restrict__ K_ref,
                            const int n_elem,
                            const int ndof_e)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    const int* my_eft = efts + e * ndof_e;

    double ue[MAX_NDOF_E];
    for (int i = 0; i < ndof_e; i++)
        ue[i] = u[my_eft[i]];

    double kue[MAX_NDOF_E] = {};
    for (int i = 0; i < ndof_e; i++)
        for (int j = 0; j < ndof_e; j++)
            kue[i] += K_ref[i * ndof_e + j] * ue[j];

    double ei = E_values[e];
    for (int i = 0; i < ndof_e; i++)
        atomicAdd(&Ku[my_eft[i]], ei * kue[i]);
}

// Accumulate diagonal of K without forming the full matrix.
// K_diag [ndof]  must be zeroed before launch.
__global__ void cuda_k_diag(double*       K_diag,
                             const double* __restrict__ E_values,
                             const int*    __restrict__ efts,
                             const double* __restrict__ K_ref,
                             const int n_elem,
                             const int ndof_e)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    const int* my_eft = efts + e * ndof_e;
    double ei = E_values[e];
    for (int i = 0; i < ndof_e; i++)
        atomicAdd(&K_diag[my_eft[i]], ei * K_ref[i * ndof_e + i]);
}

} // extern "C"
