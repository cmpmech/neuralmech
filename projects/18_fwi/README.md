# Waves through a CT Scan

A scalar wave propagating through the geometry of a computed tomography slice, for
**Chapter 18 (Simulation Acceleration via GPUs)**. The time stepping is done by the
GPU finite difference solver in `cuwave`; `helper.py` builds the grid, the material
indicator from the scan, and the corner point source that all drivers share.

## Drivers

- `2D_scalar_wave_forward_B_hai.py` _needs `data/B_Hai_1.npy`_
  one forward simulation, timed, with the wave field drawn over the void
- `animate_bhai_waves.py` _needs `data/B_Hai_1.npy`_
  the same simulation recorded into animation frames
- `plate.py`
  plate with four stacked circular voids and a linear array of 32 elements on its top edge,
  full matrix capture with `cuwave`; shared by the following drivers
- `2D_fwi.py` -> `results/rgb_png/plate_{truth,fwi}.png`
  full waveform inversion with `cuwave` (Adam, every fourth element fires)
- `tfm.py`, `2D_tfm.py` (`--pixels`) -> `results/rgb_png/plate_tfm.png`
  total focusing method: delay and sum of the analytic traces, one thread per pixel
- `2D_tfm_scaling.py` (`--max-pixels`) -> `results/data/tfm_scaling_{cpu,cupy,kernel}.csv`
  time over the image size: numpy, the same code with cupy, and the kernel
