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
