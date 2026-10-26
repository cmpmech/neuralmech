# Homogenization with Fast Fourier Transforms on the GPU

Effective stiffness of a periodic 2D cell of stiff circular fibers (plane strain) for
**Chapter 18 (Simulation Acceleration via GPUs)**, three unit load cases batched. Run from
this directory with `PYTHONPATH=../..`.

- `fft_homogenization.py`
  `Homogenization` (numpy or cupy): Moulinec-Suquet fixed point (`basic`) and conjugate
  gradients on the projected equilibrium (`cg`, Zeman et al. 2010); `FusedHomogenization`:
  the same CG with fused material, projection and update kernels in float32
- `2D_fft_homogenization.py` (`--n`) -> `results/rgb_png/fft_homogenization.png`
  strain `eps_xx`, effective stiffness and time
- `2D_fft_scaling.py` (`--max-n`) -> `results/data/fft_scaling_{cpu,gpu,fused}.csv`
  time over the resolution: numpy (float64), the same code with cupy (float64), fused (float32)

## Technicalities

The Nyquist frequencies of the projection are set to zero. Otherwise the projection is not
symmetric and CG diverges in float32 on grids of 1024 pixels and more.
