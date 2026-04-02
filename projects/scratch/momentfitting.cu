/*
 * momentfitting.cu
 * ================
 * Accumulates moment-fitting weights for one element:
 *
 *   moments[lmn] += alpha_w[ijk] * prod_d( L[d][ ijk_d * stride + lmn_d ] )
 *
 * The CPU pre-computes:
 *   lag_eval  — Lagrange basis values at all voxel IPs, concatenated per axis.
 *               Layout: [ L_0[n_vox_0, stride] | L_1[n_vox_1, stride] | ... ]
 *               Axis d starts at offset  sum_{k<d} n_vox_k * stride.
 *               Element access: lag_eval[ offset_d + j * stride + i ]
 *
 *   alpha_w   — alpha(rst_ijk) * w_ijk * voxel_detJ  for every voxel IP.
 *               Flat tensor-product order, last axis varies fastest.
 *
 * One CUDA thread owns one output moment index (lmn) and loops over all
 * voxel IPs. Parallelism across elements is handled by the caller.
 */

extern "C" __global__
void accumulate_moments(
    const double* __restrict__ lag_eval,  // concatenated per-axis blocks
    const double* __restrict__ alpha_w,   // [n_vox_total]
    double*       __restrict__ moments,   // [stride^D]  (output)
    const int D,
    const int stride,
    const int n_vox_0,
    const int n_vox_1,
    const int n_vox_2                     // ignored when D==2
)
{
    const int n_moments = (D == 2) ? stride * stride : stride * stride * stride;
    const int out_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= n_moments) return;

    const int n_vox[3] = { n_vox_0, n_vox_1, n_vox_2 };

    // Per-axis start offset into lag_eval
    int lag_offset[3];
    lag_offset[0] = 0;
    lag_offset[1] = n_vox_0 * stride;
    lag_offset[2] = (n_vox_0 + n_vox_1) * stride;

    const int n_vox_total = (D == 2) ? n_vox_0 * n_vox_1
                                     : n_vox_0 * n_vox_1 * n_vox_2;

    // Decode output flat index -> lmn multi-index
    int lmn[3] = {0, 0, 0};
    {
        int tmp = out_idx;
        if (D == 3) { lmn[2] = tmp % stride; tmp /= stride; }
        lmn[1] = tmp % stride;
        lmn[0] = tmp / stride;
    }

    double sum = 0.0;

    for (int flat_ip = 0; flat_ip < n_vox_total; ++flat_ip)
    {
        // Decode voxel flat index -> ijk multi-index (last axis fastest)
        int ijk[3] = {0, 0, 0};
        {
            int tmp = flat_ip;
            if (D == 3) { ijk[2] = tmp % n_vox[2]; tmp /= n_vox[2]; }
            ijk[1] = tmp % n_vox[1];
            ijk[0] = tmp / n_vox[1];
        }

        double lag_prod = 1.0;
        for (int d = 0; d < D; ++d)
        {
            lag_prod *= lag_eval[ lag_offset[d] + ijk[d] * stride + lmn[d] ];
        }

        sum += alpha_w[flat_ip] * lag_prod;
    }

    moments[out_idx] = sum;
}