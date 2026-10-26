# Finite Difference Waves on the GPU

One explicit finite difference step of the 2D scalar wave equation, written eight ways,
for **Chapter 18 (Simulation Acceleration via GPUs)**. Every 2D driver solves the same
problem, a gaussian pulse on a 1600 x 3200 grid with homogeneous Dirichlet
boundaries, and reports throughput in degrees of freedom per second, so the
implementations can be read as a single optimization sequence. A 1D version
introduces the halo exchange between several GPUs.

The CUDA kernels live in `step2D_wave.cu` (`basic_step` to `temporal_step`), compiled at
startup with `--use_fast_math`.

## Drivers

- `2D_wave_cupy.py`
  the step written in array slices, which cupy turns into several elementwise
  kernels; the baseline the hand-written kernels are measured against
- `2D_wave_cuda_basic.py`
  `basic_step`, one thread per grid point, with the thread index running along rows
- `2D_wave_cuda_coalesced.py`
  `coalesced_step`, the same kernel with the fast index along columns, so neighbouring
  threads read neighbouring addresses
- `2D_wave_cuda_padding.py`
  `padded_step`, row stride padded to a multiple of the warp size so every row starts
  on an aligned address
- `2D_wave_cuda_shared.py`
  `shared_step`, the block's stencil loaded into shared memory once instead of read
  from global memory five times
- `2D_wave_cuda_mixed.py`
  `mixed_step`, the field stored in half precision while the arithmetic stays single;
  each thread updates a pair of neighbouring points with 4-byte loads and stores
- `2D_wave_cuda_scaled.py` -> `results/rgb_pdf/wave_{scaled,naive}*.pdf`
  `scaled_step`, 16-bit integers with one single-precision scale per block of 32 points
  instead of half precision; fields and errors against single precision at t = 1 and 3
- `2D_wave_cuda_graph.py`
  `padded_step` on a small grid, with 20 steps captured in a CUDA graph and replayed,
  so the launch overhead is paid once per graph instead of once per step (also timed without the graph)
- `2D_wave_cuda_temporal.py`
  `temporal_step` on an 8000 x 16000 grid beyond the L2 cache; each block advances a tile
  in shared memory by eight steps per launch, bit-identical to `padded_step`
- `2D_wave_cuda_multi_gpu.py`
  the CUDA graph of `padded_step` on several GPUs; each holds a slab of rows with one
  halo row per step of the graph, exchanged with `memcpyPeer` after every replay
- `1D_wave_cuda_multi_gpu.py`
  `step1D` from `step1D_wave.cu` on several GPUs; each holds a chunk of the grid with
  one halo point per side, exchanged with `memcpyPeer` after every step
- `2D_wave_scaling.py` -> `results/data/wave_scaling_<version>.csv`, `wave_scaling_temporal.csv`
  throughput of the single precision, the mixed precision and the temporal blocking
  kernel over grid size, all replayed as CUDA graphs
- `2D_wave_scaling_multi_gpu.py` -> `results/data/wave_scaling_multi_<gpus>.csv`
  the same sweep for the multi-GPU driver on all visible GPUs
- `animate_2D_waves.py` -> `results/animations/simple_wave/`
  frames of the propagating pulse

## Roofline

- `2D_wave_roofline.py` -> `results/data/wave_roofline.csv`, `wave_roofline_ceilings.csv`
  measures the memory-bandwidth and the single-precision compute ceiling of the device,
  then the achieved Gflop/s and the arithmetic intensity of `basic_step` to `mixed_step`
