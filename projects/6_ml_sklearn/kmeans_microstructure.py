import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from postprocessing import load_cmap, save_csv

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "../18_simple_elastic_fem"))  # element stiffness matrix
sys.path.append(str(BASE_DIR / "../18_fft_homogenization"))  # fiber geometry

from fem import element_stiffness
from fft_homogenization import fibers

RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RESOLUTION = 256
CLUSTERS = [4, 8, 16, 32]
CLUSTERS_2D = [4, 16]  # shown in the principal component plane
SAMPLES_2D = 3000  # pixels drawn in the principal component plane
COMPONENTS = 3  # of the gaussian mixture

# physics: stiff fibers in a matrix, plane stress
E_FIBER, E_MATRIX, NU = 20.0, 1.0, 0.3

# ------------------------------------ create data ------------------------------------
phase = fibers(RESOLUTION)
E = np.where(phase > 0, E_FIBER, E_MATRIX).flatten()

# one bilinear element per pixel, nodes on opposite edges coincide (periodic cell)
i, j = np.divmod(np.arange(RESOLUTION**2), RESOLUTION)
corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
nodes = np.stack([(i + a) % RESOLUTION * RESOLUTION + (j + b) % RESOLUTION for a, b in corners], axis=1)
dofs = (2 * nodes[:, :, None] + np.arange(2)).reshape(-1, 8)

Ke = element_stiffness(NU)
rows = np.repeat(dofs, 8, axis=1).flatten()
cols = np.tile(dofs, 8).flatten()
K = sp.csc_matrix(((E[:, None, None] * Ke).flatten(), (rows, cols)))

# unit average strains eps_11, eps_22, eps_12 as affine displacements of the element corners
average = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [1, 0]]])
u_average = (corners @ average).reshape(3, 8)  # corners in units of the pixel size
f = -(E[:, None, None] * (u_average @ Ke)[None]).transpose(1, 0, 2)
f = np.stack([np.bincount(dofs.flatten(), load.flatten(), minlength=K.shape[0]) for load in f])

lu = splu(K[2:, 2:])  # first node fixed against rigid translation
u = np.zeros_like(f)
u[:, 2:] = lu.solve(f[:, 2:].T).T

# strains at the element centers, 3 load cases x 3 components per pixel
u_elements = u[:, dofs] + u_average[:, None]
dx = np.array([-1, 1, 1, -1]) / 2
dy = np.array([-1, -1, 1, 1]) / 2
eps = np.stack([
    u_elements[..., 0::2] @ dx,
    u_elements[..., 1::2] @ dy,
    0.5 * (u_elements[..., 0::2] @ dy + u_elements[..., 1::2] @ dx),
], axis=1)
X = eps.reshape(9, -1).T
strain = X[:, 0]  # eps_11 of the load case eps_11 = 1

# ------------------------------------- clustering ------------------------------------
labels, centers = {}, {}
for clusters in CLUSTERS:
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit(X)
    order = np.argsort(np.argsort(kmeans.cluster_centers_[:, 0]))  # by mean eps_11
    labels[clusters] = order[kmeans.labels_]
    centers[clusters] = kmeans.cluster_centers_[np.argsort(kmeans.cluster_centers_[:, 0])]

reconstruction = centers[CLUSTERS[-1]][labels[CLUSTERS[-1]], 0]  # mean eps_11 per cluster
error = np.linalg.norm(reconstruction - strain) / np.linalg.norm(strain)
print(f"K = {CLUSTERS[-1]}: relative error of cluster-averaged strain {error:.3f}")

pca = PCA(n_components=2).fit(X)
print(f"two principal components keep {pca.explained_variance_ratio_.sum():.3f} of variance")
rng = np.random.default_rng(0)
samples = rng.choice(len(X), SAMPLES_2D, replace=False)
Z = pca.transform(X[samples])

gmm = GaussianMixture(n_components=COMPONENTS, random_state=0).fit(strain[:, None])
x_test = np.linspace(strain.min(), strain.max(), 400)
densities = [
    w * norm.pdf(x_test, m, np.sqrt(v))
    for w, m, v in zip(gmm.weights_, gmm.means_[:, 0], gmm.covariances_.flatten())
]
print("mixture weights", gmm.weights_.round(3), "means", gmm.means_[:, 0].round(3))

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, len(CLUSTERS) + 2, figsize=(17, 3))
    limits = dict(vmin=strain.min(), vmax=strain.max())
    axes[0].imshow(strain.reshape(RESOLUTION, RESOLUTION).T, origin="lower", **limits)
    for ax, clusters in zip(axes[1:], CLUSTERS):
        ax.imshow(labels[clusters].reshape(RESOLUTION, RESOLUTION).T, origin="lower", cmap="cividis")
    axes[-1].imshow(reconstruction.reshape(RESOLUTION, RESOLUTION).T, origin="lower", **limits)
    for ax in axes:
        ax.axis("off")

    fig, axes = plt.subplots(1, len(CLUSTERS_2D), figsize=(10, 4))
    for ax, clusters in zip(axes, CLUSTERS_2D):
        ax.scatter(*Z.T, c=labels[clusters][samples], s=2, cmap="cividis")
        ax.scatter(*pca.transform(centers[clusters]).T, c="k", marker="x")

    fig, ax = plt.subplots()
    ax.hist(strain, bins=100, density=True, color="r")
    ax.plot(x_test, np.sum(densities, axis=0), "k")
    for density in densities:
        ax.plot(x_test, density, "b")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")
    limits = (strain.min(), strain.max())
    fields = [("strain", strain, rainbow, limits), ("reconstruction", reconstruction, rainbow, limits)]
    fields += [(f"{c}", labels[c], "cividis", (0, c - 1)) for c in CLUSTERS]
    for name, field, cmap, (vmin, vmax) in fields:
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
        ax.imshow(field.reshape(RESOLUTION, RESOLUTION).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PDF_DIR / f"kmeans_microstructure_{name}.pdf")
        plt.close(fig)

    for clusters in CLUSTERS_2D:
        save_csv(CSV_DIR / f"kmeans_pca_{clusters}.csv", z1=Z[:, 0], z2=Z[:, 1], c=labels[clusters][samples])
        z_centers = pca.transform(centers[clusters])
        save_csv(CSV_DIR / f"kmeans_pca_{clusters}_centers.csv", z1=z_centers[:, 0], z2=z_centers[:, 1])

    counts, edges = np.histogram(strain, bins=np.linspace(0, 3, 121), density=True)
    counts = np.append(counts, counts[-1])  # pgfplots ybar interval reads left edges
    save_csv(CSV_DIR / "gmm_strain_histogram.csv", x=edges, y=counts)
    components = {f"p{i}": density for i, density in enumerate(densities)}
    mixture = np.sum(densities, axis=0)
    save_csv(CSV_DIR / "gmm_strain_density.csv", x=x_test, p=mixture, **components)
