# Machine Learning Algorithms with scikit-learn

Classical machine learning algorithms for **Chapter 6 (Machine Learning Algorithms)**,
called from scikit-learn and umap-learn.

## Drivers

- `dim_reduction_shapes.py`
  six classes of binary shape images (`data/shapes_*_128.npy`) embedded in two dimensions
  by principal component analysis and by UMAP
- `dim_reduction_2D.py`
  families of curves reduced to two dimensions by SVD, PCA or UMAP (`METHOD`)
- `kmeans_microstructure.py`
  k-means clustering of the pixels of a fiber composite by their strain response to three
  unit load cases (periodic finite elements, element matrix of `18_simple_elastic_fem`),
  the clusters in the plane of the first two principal components, and a Gaussian mixture
  fitted to the strain distribution
- `knn_damage.py`
  damage location in a cantilever from the shifts of its first three eigenfrequencies
  (beam elements of `10_rom`), by k-nearest neighbors regression
- `svm_kernel.py`
  support vector machines on samples of one class inside a circle, linear on the lifted
  features $(x_1, x_2, x_1^2 + x_2^2)$ and with a radial basis function kernel
- `concrete_data.py` -> `data/concrete.npz`
  compressive strength of 1030 concrete mixtures from the UCI repository (Yeh 1998)
- `concrete_strength.py`
  linear regression, a neural network, a random forest and gradient boosting on the
  concrete data, with the permutation importance of the features; _needs concrete.npz_
- `decision_tree_regression.py`
  decision trees of increasing depth on a two-dimensional sine function
- `gradient_boosting.py`
  gradient boosting of depth-2 trees on the same function, with an increasing number of
  trees
- `random_forest_classification.py`
  single decision tree versus a random forest on the two moons dataset

## Non-obvious technicalities (authored by Claude)

In `kmeans_microstructure.py` each pixel is described by the nine strain components it
develops under the three unit macroscopic strains (its strain concentration tensor), so
pixels that deform alike end up in one cluster, wherever they lie in the cell. This is the
offline stage of self-consistent clustering analysis, where each cluster then carries a
single strain in a reduced homogenization problem. The cluster indices returned by
k-means are arbitrary; they are renumbered by the mean $\varepsilon_{xx}$ of their center,
so the colormap reads from low to high strain.

The cell is solved with one bilinear plane-stress element per pixel. The displacement is
split into the affine part $\bar{\boldsymbol{\varepsilon}}\mathbf{x}$ and a periodic
fluctuation $\tilde{\mathbf{u}}$, which is the unknown. Periodicity costs nothing: the
nodes on opposite edges get the same number (`% RESOLUTION` in the element node table), so
the grid has $N^2$ nodes instead of $(N+1)^2$. The affine part enters as the load
$-\sum_e E_e \mathbf{K}_e \mathbf{u}_{\textrm{avg}}$, where the element corners are
taken in units of the pixel size, so the strains come out without a factor $1/h$. The
periodic stiffness matrix is singular for rigid translations; fixing the first node
removes them, and one LU factorization serves all three load cases. The mean of the
fluctuation strain vanishes on a periodic mesh, so every load case averages exactly to its
unit strain.

In `knn_damage.py` the frequency shifts $\delta_i = 1 - \omega_i/\omega_{0,i}$ are divided by
their norm. To first order, a stiffness loss in one element shifts mode $i$ in proportion
to the severity times the strain energy of the mode in that element (Cawley and Adams
1979), so the normalized vector depends on the location only. Near the free end, the
curvature of all modes vanishes, the shifts fall below the measurement noise, and the
normalized vector is noise; the test error there is large for any model. The damaged
element is drawn from 50 discrete positions, so all samples of one element collapse onto
one point of the curve up to noise, and the nearest neighbors usually share the element.
