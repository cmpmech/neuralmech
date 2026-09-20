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
