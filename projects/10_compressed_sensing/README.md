# Compressed Sensing

Sparse recovery from few measurements for **Chapter 10 (Machine Learning in
Computational Mechanics)**: the minimum-norm solution of the pseudoinverse against the
$L^1$-regularized one, both in the discrete cosine basis.

## Drivers

- `compressed_sensing.py`
  a two-tone signal recovered from `SAMPLES` random time samples, the error over the
  number of samples, and a photograph recovered from `KEPT` of its pixels

## Non-obvious technicalities (authored by Claude)

The signal is sampled at the midpoints $t_n=(n+1/2)/N$. There, $\cos(2\pi f t_n)$ is
exactly the orthonormal DCT-II mode $k=2f$, so the signal is exactly 2-sparse and the
recovered spectrum shows two clean spikes; at $t_n=n/N$ it leaks into neighboring modes.

Both operators are rows of an orthonormal matrix, $\boldsymbol{\Theta}\boldsymbol{\Theta}^\top=\mathbf{I}$.
Two consequences: (1) the pseudoinverse is the transpose, so the minimum-norm solution
keeps the measured samples and puts zeros everywhere else (for the image, the
zero-filled picture); (2) the largest eigenvalue of $\boldsymbol{\Theta}^\top\boldsymbol{\Theta}$ is
one, so ISTA runs with unit step size. Plain ISTA converges slowly for a small
`REGULARIZATION`: at 1 000 iterations the 128-sample error is still 0.43, at 20 000 it
is $7\cdot10^{-3}$. The remaining error is the shrinkage bias of the soft threshold.
