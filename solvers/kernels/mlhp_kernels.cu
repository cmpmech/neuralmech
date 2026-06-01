// compile with -DUSE_FLOAT for single precision; default is double.
#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

#define MAX_NDOF_E 256 // p=3 in 3D: 4^3*3=192
#define MAX_N_SUB 512  // SUB_VOXELS=8 in 3D: 8^3=512 (3D)

// uint8 to float/double conversion (of voxel data)
__device__ __forceinline__ real_t elem_stiffness(unsigned char ind, real_t E,
                                                 real_t alpha) {
  real_t rho = (real_t)ind / (real_t)255;
  return E * (rho > alpha ? rho : alpha);
}

extern "C" {

// ------------------------------------ voxel fem

// one thread per element
__global__ void Ku_kernel(const real_t *__restrict__ u, real_t *Ku,
                          const unsigned char *__restrict__ indicator,
                          const int *__restrict__ efts,
                          const real_t *__restrict__ K_ref, const real_t E,
                          const real_t alpha, const int n_elem,
                          const int ndof_e) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem)
    return;

  const int *my_eft = efts + e * ndof_e;

  real_t ue[MAX_NDOF_E]; // needs to be known at compile-time
  for (int i = 0; i < ndof_e; i++)
    ue[i] = u[my_eft[i]];

  real_t kue[MAX_NDOF_E] = {}; // zero init
  for (int i = 0; i < ndof_e; i++)
    for (int j = 0; j < ndof_e; j++)
      kue[i] += K_ref[i * ndof_e + j] * ue[j];

  real_t Ei = elem_stiffness(indicator[e], E, alpha);
  for (int i = 0; i < ndof_e; i++)
    atomicAdd(&Ku[my_eft[i]], Ei * kue[i]);
}

// one thread per element
// accumulate diagonal of K without forming the full matrix.
__global__ void K_diag_kernel(real_t *K_diag,
                              const unsigned char *__restrict__ indicator,
                              const int *__restrict__ efts,
                              const real_t *__restrict__ K_ref, const real_t E,
                              const real_t alpha, const int n_elem,
                              const int ndof_e) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem)
    return;

  const int *my_eft = efts + e * ndof_e;
  real_t Ei = elem_stiffness(indicator[e], E, alpha);
  for (int i = 0; i < ndof_e; i++)
    atomicAdd(&K_diag[my_eft[i]], Ei * K_ref[i * ndof_e + i]);
}

// ------------------------- multiresolution fem (subvoxels)

// one thread per element
__global__ void assemble_K_e_kernel(real_t *K_e,
                                    const unsigned char *__restrict__ indicator,
                                    const real_t *__restrict__ K_refs,
                                    const real_t E, const real_t alpha,
                                    const int n_elem, const int ndof_e,
                                    const int n_sub, const int S, const int Sz,
                                    const int Ny_elem, const int Nz_elem,
                                    const int Ny_vox, const int Nz_vox) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem)
    return;

  // element indices
  int ix_e = e / (Ny_elem * Nz_elem);
  int iy_e = (e / Nz_elem) % Ny_elem;
  int iz_e = e % Nz_elem;

  // pass 1: per-subvoxel material scaling
  real_t Ei[MAX_N_SUB];
  for (int s = 0; s < n_sub; s++) {
    // local subvoxel indices (within element)
    int sx = s / (S * Sz);
    int sy = (s / Sz) % S;
    int sz = s % Sz;
    // global voxel indices
    int ix_v = ix_e * S + sx;
    int iy_v = iy_e * S + sy;
    int iz_v = iz_e * Sz + sz;
    // global voxel index in C ordering
    int vox = ix_v * (Ny_vox * Nz_vox) + iy_v * Nz_vox + iz_v;
    Ei[s] = elem_stiffness(indicator[vox], E, alpha);
  }

  // pass 2: accumulate into K_e
  real_t *my_K_e = K_e + e * ndof_e * ndof_e; // helper pointer
  for (int k = 0; k < ndof_e * ndof_e; k++) { // over K_e_cur entries
    real_t acc = (real_t)0;                   // initialize contribution
    for (int s = 0; s < n_sub; s++)           // over each voxel
      acc += Ei[s] * K_refs[s * ndof_e * ndof_e + k];
    my_K_e[k] = acc;
  }
}

// one thread per element
__global__ void Ku_subvoxel_kernel(const real_t *__restrict__ u, real_t *Ku,
                                   const int *__restrict__ efts,
                                   const real_t *__restrict__ K_e,
                                   const int n_elem, const int ndof_e) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem)
    return;

  const int *my_eft = efts + e * ndof_e;
  const real_t *my_K = K_e + e * ndof_e * ndof_e;

  real_t ue[MAX_NDOF_E];
  for (int i = 0; i < ndof_e; i++)
    ue[i] = u[my_eft[i]];

  real_t kue[MAX_NDOF_E] = {};
  for (int i = 0; i < ndof_e; i++)
    for (int j = 0; j < ndof_e; j++)
      kue[i] += my_K[i * ndof_e + j] * ue[j];

  for (int i = 0; i < ndof_e; i++)
    atomicAdd(&Ku[my_eft[i]], kue[i]);
}

// one thread per element
// accumulate diagonal of K without forming the full matrix.
__global__ void K_diag_subvoxel_kernel(real_t *K_diag,
                                       const int *__restrict__ efts,
                                       const real_t *__restrict__ K_e,
                                       const int n_elem, const int ndof_e) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem)
    return;

  const int *my_eft = efts + e * ndof_e;
  const real_t *my_K = K_e + e * ndof_e * ndof_e;
  for (int i = 0; i < ndof_e; i++)
    atomicAdd(&K_diag[my_eft[i]], my_K[i * ndof_e + i]);
}
}
