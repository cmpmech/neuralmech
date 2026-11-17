# Fourier Analysis

Fourier series and transforms for **Chapter 10 (Machine Learning in Computational
Mechanics)**: a 1D series, and 2D transforms of the grayscale photographs in
`data/images/`.

## Drivers

- `fourier_series.py`
  truncated Fourier series of a smooth function and of a square wave, with the decay of
  their coefficients and the Gibbs overshoot at the jump
- `fft2.py`
  magnitude and phase spectrum of two photographs, and the images reconstructed from the
  largest `KEEP_RATIOS` of the coefficients (lossy compression)
- `fft2_denoise.py`
  additive gaussian noise removed by keeping only the coefficients within `RADIUS`
  of the spectrum center (low-pass filtering)
- `basis_comparison.py`
  relative error over storage for the truncated SVD (a basis computed from the image),
  the largest 2D FFT coefficients, and the largest coefficients of an 8x8 block DCT
  (JPEG-like)
- `stft_damage.py`
  spectrogram of the top-floor acceleration of a three-story frame under random excitation
  whose first story softens during the record, next to the spectrum of the whole record

## Non-obvious technicalities (authored by Claude)

`fourier_series.py` samples at the midpoints $x_n=(n+1/2)/N$. On these points the
sampled square wave is exactly odd about $x=1/2$, so its even harmonics vanish to round-off
as in the continuous series; sampling at $x_n=n/N$ puts $\mathrm{sign}(0)=0$ and
$\mathrm{sign}(\sin\pi)=+1$ on the grid and leaks spurious even harmonics of order $10^{-3}$.

`basis_comparison.py` counts storage in stored real numbers, as a fraction of the
$H\cdot W$ pixels. A rank-$R$ SVD stores $R(H+W+1)$ reals. A real image has a
conjugate-symmetric spectrum, $\hat{f}_{-k}=\overline{\hat{f}_k}$; the two members of a
pair share their magnitude, are kept or dropped together, and are stored as one complex
number, so on average one kept FFT coefficient costs one real. A DCT coefficient is real
and costs one. The positions of the kept FFT and DCT coefficients are not counted (JPEG
encodes them implicitly by run lengths along a zigzag order), which favors the two fixed
bases. On `flying_goose.jpg` the SVD is the worst of the three at every storage level: its
basis is specific to the image and has to be stored along with the coefficients, whereas
the Fourier and cosine bases are known in advance. The block DCT wins above roughly 2 %
storage, the global FFT below, where less than one coefficient per 8x8 block remains.
