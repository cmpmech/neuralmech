# Machine Learning Algorithms from Scratch

Classical machine learning algorithms for **Chapter 6 (Machine Learning Algorithms)**,
written out directly instead of called from a library.

## Drivers

- `svd.py`
  singular value decomposition implemented from the eigendecomposition of $AA^\intercal$,
  applied to a grayscale photograph, whose truncation is a low-rank image compression
- `pca.py`
  the same truncation with the row mean removed first, so the retained directions are
  the principal components, with the decay of the explained variance
- `pca_2D.py`
  dimensionality reduction of a library of function families onto two principal
  components, where most families separate into distinct clusters but two of them do not

`RANKS` sets which truncations the two image drivers show; `FUNCTIONS` is the library
`pca_2D.py` samples. The preview shows the reconstructions side by side and the spectrum
in a second window, `--book` exports the data as a CSV.

## Non-obvious technicalities (authored by Claude)

The singular value decomposition in `svd.py` is obtained from the symmetric eigenvalue
problem $AA^\intercal = U\Sigma^2U^\intercal$, after which the right singular vectors
follow from $V^\intercal = \Sigma^{-1}U^\intercal A$. Deriving $V$ this way, rather than
from a second eigenvalue problem for $A^\intercal A$, keeps the signs of the two factors
consistent by construction and solves only the smaller of the two eigenvalue problems.
The price is that forming the Gram matrix squares the condition number, so the smallest
singular values lose accuracy -- irrelevant for a truncation, which keeps the largest.

The centering is the only difference between `pca.py` and `svd.py`, and on a single
photograph it is close to meaningless: the rows of an image are not samples drawn from a
population. Both drivers therefore produce almost the same pictures. Centering earns its
keep in `pca_2D.py`, where each sample is one curve and the mean curve is a meaningful
quantity. Without it the leading component is spent almost entirely on the mean, and a
third component is needed to see what two centered components already show.

The last two families in `pca_2D.py` are the instructive case. `sine_phase` draws a sine
of fixed amplitude and random phase, `sine_mix` an independent combination of a sine and
a cosine of the same frequency. Both span the same two-dimensional subspace, so the
reduction maps them onto the same plane: the first lands on a circle, the second fills
the disk enclosed by it. The two are perfectly distinguishable -- by the radius -- but
no linear method separates them, and a projection cannot undo that. The remaining
families differ in the directions the projection does retain and separate cleanly.
