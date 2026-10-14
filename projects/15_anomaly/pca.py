"""PCA scatter of latent means (mu) for the normal and anomaly sets.

python projects/15_anomaly/fiber_vae_pca.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------ config (edit) ------------------------------
# MODEL = (BASE_DIR / "../../models/_latent20_beta0.03_256.pt2").resolve()
MODEL = (BASE_DIR / "../../models/fiber_vae_depth5_latent20_beta0.001_256.pt2").resolve()
DATA = (BASE_DIR / "../../data").resolve()
# SETS = {"normal": DATA / "t_0sq_2tot_800.npy", "square": DATA / "t_1sq_2tot_800.npy"}
SETS = {                                           # name: (file, label) label 0 = normal, 1 = anomaly
    "normal_paired": DATA / "fibers_paired_700_r0.09_0.09_seed33.npy",    # must be disjoint from FIT (kNN would match itself)
    "square_paired": DATA / "fibers_paired_700_r0.09_0.09_seed33_1anom.npy",
}
FIT_ON = None        # None: PCA basis from all sets pooled; or a set name, e.g. "normal"


@torch.no_grad()
def latent_means(model, path, bs=256):
    x = torch.from_numpy(np.load(path)).float().unsqueeze(1)    # raw [0,1], inactive standardizer
    return torch.cat([model(xb.to(device))[1].cpu() for xb in x.split(bs)])   # model -> (x_pred, mean, logvar)


if __name__ == "__main__":
    model = torch.load(MODEL, weights_only=False, map_location=device).eval()
    mu = {name: latent_means(model, p) for name, p in SETS.items()}

    ref = torch.cat(list(mu.values())) if FIT_ON is None else mu[FIT_ON]
    m = ref.mean(0)
    _, S, Vh = torch.linalg.svd(ref - m, full_matrices=False)    # rows of Vh = principal axes
    var = (S**2 / (S**2).sum()).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (i, j) in zip(axes, [(0, 1), (2, 3)]):     # PCA directions to plot
        for name, z in mu.items():
            p = (z - m) @ Vh.T
            ax.scatter(p[:, i], p[:, j], s=6, alpha=0.4, label=name)
        ax.set_xlabel(f"PC{i + 1} ({var[i]:.0%})")
        ax.set_ylabel(f"PC{j + 1} ({var[j]:.0%})")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(BASE_DIR / "../../tmp/latent_pca.png", dpi=150)
    plt.show()