# Fourier Transforms of Images

Two-dimensional discrete Fourier transforms of photographs, supporting
**Chapter 10 (Machine Learning in Computational Mechanics)**. Both drivers read a
grayscale image from `data/images/` and work on its shifted spectrum.

## Drivers

- `fft2.py`
  magnitude and phase spectrum, and the image reconstructed from the largest
  `KEEP_RATIO` of the coefficients (lossy compression)
- `fft2_denoise.py`
  additive gaussian noise removed by keeping only the coefficients within `RADIUS`
  of the spectrum center (low-pass filtering)
