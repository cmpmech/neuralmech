# Machine Learning Algorithms from Scratch

Classical machine learning algorithms for **Chapter 6 (Machine Learning Algorithms)**,
written out directly instead of called from a library.

## Drivers

- `svd.py`
  singular value decomposition implemented from the eigendecomposition of $AA^\intercal$,
  applied to a grayscale photograph (`data/images/sheep.jpg`), whose truncation is a
  low-rank image compression, and its first rank-one terms with their column and row
  profiles
- `pca.py`
  the same truncation with the row mean removed first, so the retained directions are
  the principal components, with the decay of the explained variance
- `pca_eigenfaces.py`
  eigenfaces of the Olivetti faces (scikit-learn, cached in `data/olivetti`), with one
  person held out and reconstructed from an increasing number of components
- `svm_yield_surface.py`
  linear support vector machine trained by subgradient descent on the hinge loss, once on
  plane stress states and once on their quadratic features, learning the von Mises yield
  surface from elastic/plastic labels
- `gaussian_process.py`
  Gaussian process regression with a squared exponential kernel, hyperparameters from the
  marginal likelihood, on the sine data of Chapter 5

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
population. Both drivers therefore produce almost the same pictures.

In `svm_yield_surface.py` the quadratic features $(\sigma_1^2, \sigma_2^2, \sigma_1\sigma_2)$
make the von Mises criterion $\sigma_1^2 - \sigma_1\sigma_2 + \sigma_2^2 \le \sigma_y^2$
linear in the features, so a linear classifier in the lifted space recovers it. Dividing
the learned weights by $-b$ normalizes the boundary to the form
$a_1\sigma_1^2 + a_2\sigma_2^2 + a_3\sigma_1\sigma_2 = 1$, whose coefficients approach
$(1, 1, -1)$. They are not exact, since the hinge loss places the boundary in the middle
of the gap between the nearest samples of the two classes, not on the true surface. The
support vectors are marked as the samples with $y f(\mathbf{x}) \approx 1$.
