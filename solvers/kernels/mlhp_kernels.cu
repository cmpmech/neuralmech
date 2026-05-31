// Compile with -DUSE_FLOAT for single precision; default is double.
#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

#define MAX_NDOF_E 256 // p=3 in 3D: 4^3*3=192

// uint8 to float/double conversion (of voxel data)
__device__ __forceinline__ real_t elem_stiffness(unsigned char ind, real_t E,
                                                 real_t alpha) {
  real_t rho = (real_t)ind / (real_t)255;
  return E * (rho > alpha ? rho : alpha);
}

extern "C" {

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

// ------------------------------------------------------------------------------------
//
// --------------- subvoxel kernels (pre-assembled K_e per element)
// -----------------
//
// Each macro-element covers S^D voxels.  K_refs[n_sub, ndof_e, ndof_e] holds
// one reference matrix per subvoxel (geometry-only, E=1).  cuda_assemble_K_e
// builds K_e[e] = sum_s  E * max(rho_s, alpha) * K_refs[s]  once before the CG
// loop. cuda_matvec_elem and cuda_k_diag_elem then operate on the pre-assembled
// K_e.
//
// Subvoxel index s encodes position in C-order: s = sx*(S*Sz) + sy*Sz + sz
//   2D: Sz=1  →  s = sx*S + sy
//   3D: Sz=S  →  s = sx*S^2 + sy*S + sz
// Element index e encodes grid position identically (with elem grid dims).

#define MAX_N_SUB 512 // SUB_VOXELS^D: 8^3=512, 6^3=216

// Assemble K_e[e] from K_refs and indicator.  One thread per element.
// Scalar-accumulator pattern: precompute ei[s], then outer loop over the
// ndof_e^2 output entries with a register accumulator — K_e[e] is written
// exactly once, eliminating the read-modify-write traffic of an outer-s loop.
__global__ void
cuda_assemble_K_e(real_t *K_e, const unsigned char *__restrict__ indicator,
                  const real_t *__restrict__ K_refs, const real_t E,
                  const real_t alpha, const int n_elem, const int ndof_e,
                  const int n_sub, const int S, const int Sz, const int Ny_elem,
                  const int Nz_elem, const int Ny_vox, const int Nz_vox) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_elem)
    return;

  int ix_e = e / (Ny_elem * Nz_elem);
  int iy_e = (e / Nz_elem) % Ny_elem;
  int iz_e = e % Nz_elem;

  // Pass 1: per-subvoxel material scaling (n_sub coalesced indicator reads).
  real_t ei[MAX_N_SUB];
  for (int s = 0; s < n_sub; s++) {
    int sx = s / (S * Sz);
    int sy = (s / Sz) % S;
    int sz = s % Sz;
    int ix_v = ix_e * S + sx;
    int iy_v = iy_e * S + sy;
    int iz_v = iz_e * Sz + sz;
    int vox = ix_v * (Ny_vox * Nz_vox) + iy_v * Nz_vox + iz_v;
    real_t rho = (real_t)indicator[vox] / (real_t)255;
    ei[s] = E * (rho > alpha ? rho : alpha);
  }

  // Pass 2: accumulate into register, write K_e_out[k] once.
  // K_refs[s * ndof_e2 + k] is the same address for all threads in a warp
  // (same s and k), so it broadcasts from L2 (~2.4 MB total, fits in cache).
  int ndof_e2 = ndof_e * ndof_e;
  real_t *K_e_out = K_e + e * ndof_e2;
  for (int k = 0; k < ndof_e2; k++) {
    real_t acc = (real_t)0;
    for (int s = 0; s < n_sub; s++)
      acc += ei[s] * K_refs[s * ndof_e2 + k];
    K_e_out[k] = acc;
  }
}

// Matrix-vector product using pre-assembled K_e.  One thread per element.
// Ku must be zeroed before launch.
__global__ void cuda_matvec_elem(const real_t *__restrict__ u, real_t *Ku,
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

// Accumulate diagonal of pre-assembled K_e into K_diag.  One thread per
// element. K_diag must be zeroed before launch.
__global__ void cuda_k_diag_elem(real_t *K_diag, const int *__restrict__ efts,
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
