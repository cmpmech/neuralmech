# Lattice Boltzmann on the GPU

One D2Q9 lattice Boltzmann step with two-relaxation-time collision, written seven ways, for
**Chapter 18 (Simulation Acceleration via GPUs)**. Every driver solves the same problem,
decaying turbulence from random vortices in a periodic rectangle of 8192 x 4096 nodes, and reports
throughput in node updates per second, so the implementations read as one optimization sequence.

The CUDA kernels live in `lbm.cu` (`collide_aos` to `temporal_step`), the lattice tables, the
random initial field and the diagnostics in `lbm.py`.

## Drivers

- `2D_lbm_cupy.py`
  the step written in array operations, streaming by `cp.roll`; the baseline
- `2D_lbm_cuda_basic.py`
  `collide_aos` and `stream_aos`, one thread per node, the nine populations of a node side by
  side (array of structures), push streaming in a second kernel
- `2D_lbm_cuda_soa.py`
  `collide_soa` and `stream_soa`, one array per population (structure of arrays)
- `2D_lbm_cuda_fused.py`
  `fused_step`, streaming and collision in one kernel that pulls from the neighbors
- `2D_lbm_cuda_mixed.py`
  `fused_step` with the populations stored in half precision; deviation from single precision
- `2D_lbm_cuda_inplace.py`
  `aa_even` and `aa_odd`, the AA pattern with a single array in half precision; bit-identical
  to `fused_step`
- `2D_lbm_cuda_temporal.py`
  `temporal_step`, six steps per launch on tiles of 48 x 64 nodes, populations in registers,
  moving populations exchanged through shared memory; deviation from single precision
- `2D_lbm_cuda_graph.py`
  `fused_step` on a 512 x 256 grid, single launches against a CUDA graph of 20 steps
- `2D_lbm_turbulence.py` -> `results/rgb_png/lbm_turbulence.png`, `results/animations/animation_frames/lbm_turbulence/`
  30000 steps with `temporal_step`; vorticity at the end (`--book`) and 250 frames (`--animate`)

## Extra (not in the book)

- `extra/2D_lbm_packing.py` -> `results/animations/animation_frames/lbm_packing/`
  flow through a random packing of cylinders, 16384 x 4096 nodes; `temporal_walls_step` and
  `sponge` from `extra/walls.cu` add bounce-back and an outflow sponge to the periodic grid.
  Run from `extra/` with `PYTHONPATH=..`

## Non-obvious technicalities (authored by Claude)

- **Deviation storage.** All kernels store $g_i = f_i - w_i$. The populations stay within about
  $10^{-3}$ of their weights, which half precision would round away; the deviations keep about
  three significant digits.
- **Stability.** The two-relaxation-time collision with the magic parameter $1/4$ stayed stable
  for $\tau \geq 0.524$ in all tried flows. At $\tau \leq 0.518$, under-resolved shear layers
  diverged within a few ten thousand steps, in single precision as well.
- **Temporal blocking.** Each block loads its tile with a halo of six nodes and advances it by
  six steps. Edge nodes pull garbage, which travels one node per step and never reaches the
  inner nodes that are written back. The new populations go to a second array, since neighboring
  tiles still read the halo. Single-precision storage gives results identical to `fused_step`.
- **AA pattern.** An even step collides in place and stores $g_i^*$ in slot $\bar{\imath}$; an
  odd step reads slot $\bar{\imath}$ of $x - c_i$ and writes slot $i$ of $x + c_i$. Each thread
  writes exactly the slots it reads, so a single array suffices. Only after odd steps are the
  populations in their usual slots.
