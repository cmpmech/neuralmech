# Finite Difference Waves on the GPU

One explicit finite difference step of the 2D scalar wave equation, written six ways,
for **Chapter 18 (Simulation Acceleration via GPUs)**. Every driver solves the same
problem, a gaussian pulse on a 1600 x 3200 grid with homogeneous Dirichlet
boundaries, and reports throughput in degrees of freedom per second, so the
implementations can be read as a single optimization sequence.

The CUDA kernels live in `step2D_wave.cu` (`fd_kernelV1` to `fd_kernelV5`). Set
`PRECOMPILED = True` in a driver to load `step2D_wave.ptx` instead of compiling the
source at startup.

## Drivers

- `2D_wave_cupy.py`
  the step written in array slices, which cupy turns into several elementwise
  kernels; the baseline the hand-written kernels are measured against
- `2D_wave_cuda_basic.py`
  `fd_kernelV1`, one thread per grid point, with the thread index running along rows
- `2D_wave_cuda_coalesced.py`
  `fd_kernelV2`, the same kernel with the fast index along columns, so neighbouring
  threads read neighbouring addresses
- `2D_wave_cuda_padding.py`
  `fd_kernelV3`, row stride padded to a multiple of the warp size so every row starts
  on an aligned address
- `2D_wave_cuda_shared.py`
  `fd_kernelV4`, the block's stencil loaded into shared memory once instead of read
  from global memory five times
- `2D_wave_cuda_mixed.py`
  `fd_kernelV5`, the field stored in half precision while the arithmetic stays single
- `2D_reference_warp.py`
  the same kernel written in warp, which compiles Python to CUDA
- `2D_wave_scaling_precompiled.py` -> `results/data/wave_scaling_<version>.csv`
  throughput of the single and the mixed precision kernel over grid size
- `2D_warp_scaling.py` -> `results/data/warp_scaling.csv`
  the same sweep for the warp kernel
- `animate_2D_waves.py` -> `results/animations/simple_wave/`
  frames of the propagating pulse

`forward_scaling.jl` is the julia counterpart of the scaling sweeps.
