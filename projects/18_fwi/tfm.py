"""Total focusing method: delay and sum of the full matrix capture onto an image grid.

The image at pixel p is |sum_e sum_r h_er(t0 + (|p - x_e| + |p - x_r|) / c)| with the
analytic signal h (Hilbert transform along time) of the traces, linearly interpolated.
"""

import cupy as cp
import numpy as np

# the number of elements E is a compile-time constant, so that the loops unroll and the
# times of flight stay in registers
TFM = r"""
extern "C" __global__ void tfm(const float2 *h, const float *xe, const float ye,
                               const int Nt, const float dt, const float t0, const float c,
                               const float *xp, const float *yp, const int P, float *image){
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= P) return;
    float tau[E];  // time of flight from every element
    #pragma unroll
    for(int e = 0; e < E; e++) tau[e] = hypotf(xp[p] - xe[e], yp[p] - ye) / c;
    float2 s = {0, 0};
    #pragma unroll
    for(int e = 0; e < E; e++){
        #pragma unroll
        for(int r = e; r < E; r++){  // reciprocity: h_er = h_re, the upper triangle twice
            const float t = (t0 + tau[e] + tau[r]) / dt;
            const int k = (int)t;
            if(k >= Nt - 1) continue;
            const float w = (t - k) * (r > e ? 2 : 1), v = (r > e ? 2 : 1) - w;
            const float2 a = h[(e * E + r) * Nt + k], b = h[(e * E + r) * Nt + k + 1];
            s.x += v * a.x + w * b.x;
            s.y += v * a.y + w * b.y;
        }
    }
    image[p] = hypotf(s.x, s.y);
}"""


def tfm(h, elements, t0, dt, c, x, y, xp=np):
    """the delay and sum written with array operations, for numpy or cupy alike.

    h: (E, E, Nt) complex analytic traces, elements: (E, 2), x, y: pixel coordinates
    """
    E, _, Nt = h.shape
    tau = [xp.hypot(x - ex, y - ey) / c for ex, ey in elements]
    s = xp.zeros(x.shape, h.dtype)
    for e in range(E):
        for r in range(E):
            t = (t0 + tau[e] + tau[r]) / dt
            k = xp.minimum(t.astype(np.int32), Nt - 2)
            w = t - k
            s += ((1 - w) * h[e, r, k] + w * h[e, r, k + 1]) * (k < Nt - 2)
    return xp.abs(s)


def tfm_kernel(h, elements, t0, dt, c, x, y):
    """the same with one thread per pixel, h on the device as complex64"""
    E, _, Nt = h.shape
    image = cp.empty(x.size, cp.float32)
    xe = cp.asarray(elements[:, 0], cp.float32)
    kernel = cp.RawKernel(TFM, "tfm", options=(f"-DE={E}",))
    kernel(((x.size + 127) // 128,), (128,),
           (h, xe, np.float32(elements[0, 1]), np.int32(Nt), np.float32(dt),
            np.float32(t0), np.float32(c), cp.asarray(x, cp.float32).ravel(),
            cp.asarray(y, cp.float32).ravel(), np.int32(x.size), image))
    return image.reshape(x.shape)
