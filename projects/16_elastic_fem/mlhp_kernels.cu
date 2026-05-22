// Matrix-free K*U kernels for structured voxel FEM built on mlhp.
//
// Differences from elasticity_mf.cu:
//   - EFT passed as an explicit device array (supports any polynomial degree)
//   - K_ref passed as a device pointer (not constant memory), any ndof_e
//   - No hardcoded Q1 geometry: both 2D and 3D handled identically
//
// Compile with -DUSE_FLOAT for single precision; default is double.

#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

extern "C" {

#define MAX_NDOF_E 64 // enough for p=4 in 3D (64 DOFs per element)

// One thread per element.
// u         [ndof]              input DOF vector
// Ku        [ndof]              output, must be zeroed before launch
// indicator [n_elem]            uint8 voxel density (0=void, 255=solid)
// efts      [n_elem * ndof_e]   element freedom tables (row-major)
// K_ref     [ndof_e * ndof_e]   reference element stiffness with E=1 (row-major)
__global__ void cuda_matvec(const real_t *__restrict__ u, real_t *Ku,
                            const unsigned char *__restrict__ indicator,
                            const int *__restrict__ efts,
                            const real_t *__restrict__ K_ref,
                            const real_t E, const real_t alpha,
                            const int n_elem, const int ndof_e) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem)
    return;

  const int *my_eft = efts + e * ndof_e;

  real_t ue[MAX_NDOF_E];
  for (int i = 0; i < ndof_e; i++)
    ue[i] = u[my_eft[i]];

  real_t kue[MAX_NDOF_E] = {};
  for (int i = 0; i < ndof_e; i++)
    for (int j = 0; j < ndof_e; j++)
      kue[i] += K_ref[i * ndof_e + j] * ue[j];

  real_t rho = (real_t)indicator[e] / (real_t)255;
  real_t ei = E * (rho > alpha ? rho : alpha);
  for (int i = 0; i < ndof_e; i++)
    atomicAdd(&Ku[my_eft[i]], ei * kue[i]);
}

// Accumulate diagonal of K without forming the full matrix.
// K_diag [ndof]  must be zeroed before launch.
__global__ void cuda_k_diag(real_t *K_diag,
                            const unsigned char *__restrict__ indicator,
                            const int *__restrict__ efts,
                            const real_t *__restrict__ K_ref,
                            const real_t E, const real_t alpha,
                            const int n_elem, const int ndof_e) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem)
    return;

  const int *my_eft = efts + e * ndof_e;
  real_t rho = (real_t)indicator[e] / (real_t)255;
  real_t ei = E * (rho > alpha ? rho : alpha);
  for (int i = 0; i < ndof_e; i++)
    atomicAdd(&K_diag[my_eft[i]], ei * K_ref[i * ndof_e + i]);
}

} // extern "C"
