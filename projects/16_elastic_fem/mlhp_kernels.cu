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

// ---------------------------------------------------------------------------
// Subvoxel variants: each macro-element spans sub_voxels^D voxels.
// K_refs [n_sub * ndof_e * ndof_e] — one reference matrix per sub-voxel pos.
// Indicator is the full voxel grid (Nx_vox x Ny_vox [x Nz_vox]), C-order.
// ---------------------------------------------------------------------------

__global__ void cuda_matvec_sub(
    const real_t *__restrict__ u, real_t *Ku,
    const unsigned char *__restrict__ indicator,
    const int *__restrict__ efts,
    const real_t *__restrict__ K_refs,
    const real_t E, const real_t alpha,
    const int n_elem, const int ndof_e, const int n_sub,
    const int sub_voxels, const int sub_voxels_z,
    const int Ny_elem, const int Nz_elem,
    const int Ny_vox, const int Nz_vox) {

  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem) return;

  int ez = e % Nz_elem;
  int ey = (e / Nz_elem) % Ny_elem;
  int ex = e / (Ny_elem * Nz_elem);

  const int *my_eft = efts + e * ndof_e;

  real_t ue[MAX_NDOF_E];
  for (int i = 0; i < ndof_e; i++)
    ue[i] = u[my_eft[i]];

  real_t kue[MAX_NDOF_E] = {};

  for (int s = 0; s < n_sub; s++) {
    int sk = s % sub_voxels_z;
    int sj = (s / sub_voxels_z) % sub_voxels;
    int si = s / (sub_voxels * sub_voxels_z);

    int vox_idx = (ex * sub_voxels + si) * Ny_vox * Nz_vox
                + (ey * sub_voxels + sj) * Nz_vox
                + (ez * sub_voxels_z + sk);

    real_t rho = (real_t)indicator[vox_idx] / (real_t)255;
    real_t ei = E * (rho > alpha ? rho : alpha);

    const real_t *K_ref_s = K_refs + s * ndof_e * ndof_e;
    for (int i = 0; i < ndof_e; i++)
      for (int j = 0; j < ndof_e; j++)
        kue[i] += ei * K_ref_s[i * ndof_e + j] * ue[j];
  }

  for (int i = 0; i < ndof_e; i++)
    atomicAdd(&Ku[my_eft[i]], kue[i]);
}

__global__ void cuda_k_diag_sub(
    real_t *K_diag,
    const unsigned char *__restrict__ indicator,
    const int *__restrict__ efts,
    const real_t *__restrict__ K_refs,
    const real_t E, const real_t alpha,
    const int n_elem, const int ndof_e, const int n_sub,
    const int sub_voxels, const int sub_voxels_z,
    const int Ny_elem, const int Nz_elem,
    const int Ny_vox, const int Nz_vox) {

  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem) return;

  int ez = e % Nz_elem;
  int ey = (e / Nz_elem) % Ny_elem;
  int ex = e / (Ny_elem * Nz_elem);

  const int *my_eft = efts + e * ndof_e;

  for (int s = 0; s < n_sub; s++) {
    int sk = s % sub_voxels_z;
    int sj = (s / sub_voxels_z) % sub_voxels;
    int si = s / (sub_voxels * sub_voxels_z);

    int vox_idx = (ex * sub_voxels + si) * Ny_vox * Nz_vox
                + (ey * sub_voxels + sj) * Nz_vox
                + (ez * sub_voxels_z + sk);

    real_t rho = (real_t)indicator[vox_idx] / (real_t)255;
    real_t ei = E * (rho > alpha ? rho : alpha);

    const real_t *K_ref_s = K_refs + s * ndof_e * ndof_e;
    for (int i = 0; i < ndof_e; i++)
      atomicAdd(&K_diag[my_eft[i]], ei * K_ref_s[i * ndof_e + i]);
  }
}

} // extern "C"
