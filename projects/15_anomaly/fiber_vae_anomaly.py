"""Score images for anomalies with a trained fiber VAE.

Scores (higher = more anomalous):
  bce_mean, mse_mean : image-mean reconstruction error (decode(mu), no sampling)
  maha, knn          : distance of mu to the latents of the normal "fit" set

python projects/15_anomaly/fiber_vae_anomaly.py
"""
from pathlib import Path
from tkinter import RIDGE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fiber_vae_reconstruct import plot_dataset_samples

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------ config (edit) ------------------------------
# MODEL = (BASE_DIR / "../../models/_latent20_beta0.03_256.pt2").resolve()
# MODEL = (BASE_DIR / "../../models/_fullyTrained_plateausched__latent40_beta0.08_256.pt2").resolve()
# MODEL = (BASE_DIR / "../../models/bce_dice__latent40_betaBoth0.04_256.pt2").resolve()
# MODEL = (BASE_DIR / "../../models/fiber_vae_depth5_latent20_beta0.03_256.pt2").resolve()
MODEL = (BASE_DIR / "../../models/fiber_vae_depth5_latent20_beta0.001_256.pt2").resolve()
DATA = (BASE_DIR / "../../data").resolve()
FIT = DATA / "t_0sq_2tot_800.npy"                  # normals the model trained on (fits latent stats)
SETS = {                                           # name: (file, label) label 0 = normal, 1 = anomaly
    "normal_paired": (DATA / "fibers_paired_700_r0.09_0.09_seed33.npy", 0),    # must be disjoint from FIT (kNN would match itself)
    "square": (DATA / "t_1sq_2tot_800.npy", 1),
    "square_paired": (DATA / "fibers_paired_700_r0.09_0.09_seed33_1anom.npy", 1),
    # "squares2": (DATA / "fibers_anomaly_2_256.npy", 1),
}
SCORES = ["bce_mean", "mse_mean", "maha", "knn", "reenc"]
K       = 5                     # kNN neighbors for knn score
RIDGE   = 1e-2                  # covariance ridge for mahalanobis score
FPR     = 0.1                  # False Positive Rate (FPR) for True Positive Rate(TPR) column


# --------------------------------- scoring ---------------------------------
def load(path):
    return torch.from_numpy(np.load(path)).float().unsqueeze(1)    # (N,1,H,W), raw [0,1]


@torch.no_grad()
def encode_decode(model, x, bs=256):
    mus, recs, mus2 = [], [], []
    for xb in x.split(bs):
        _, mu, _ = model(xb.to(device))
        rec = model.decode(mu)
        _, mu2, _ = model(rec)                    # re-encode the reconstruction
        mus.append(mu.cpu()); recs.append(rec.cpu()); mus2.append(mu2.cpu())
    return torch.cat(mus), torch.cat(recs), torch.cat(mus2)


class LatentScorer:
    """Distances from mu to the normal fit-set latents."""

    def __init__(self, mu_fit, k=K, ridge=RIDGE):
        self.mu_fit, self.k = mu_fit, k
        self.m = mu_fit.mean(0)
        cov = torch.cov((mu_fit - self.m).T) + ridge * torch.eye(mu_fit.shape[1])
        self.prec = torch.linalg.inv(cov)      # ridge tames collapsed (near-zero variance) dims

    def maha(self, mu):
        d = mu - self.m
        return ((d @ self.prec) * d).sum(1).sqrt()

    def knn(self, mu):
        """mean distance to the 5 nearest fit latents"""
        return torch.cdist(mu, self.mu_fit).topk(self.k, largest=False).values.mean(1)


def image_scores(x, rec):
    per_img = lambda t: t.flatten(1).mean(1)
    return {"bce_mean": per_img(F.binary_cross_entropy(rec, x, reduction="none")),
            "mse_mean": per_img(F.mse_loss(rec, x, reduction="none"))}


def score_sets(model, scorer, sets):
    """Tidy DataFrame: one row per image, one column per score."""
    frames = []
    for name, (path, label) in sets.items():
        x = load(path)
        mu, rec, mu2 = encode_decode(model, x)
        s = {**image_scores(x, rec), "maha": scorer.maha(mu), "knn": scorer.knn(mu), "reenc": (mu - mu2).norm(dim=1)}
        frames.append(pd.DataFrame({k: v.numpy() for k, v in s.items()}).assign(set=name, label=label))
    return pd.concat(frames, ignore_index=True)


# ------------------------------- evaluation --------------------------------
def auroc(y, s):
    """Mann-Whitney AUROC via ranks (ties averaged)."""
    r, n1 = s.rank(), int(y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def summarize(df, scores=SCORES, fpr=FPR):
    """AUROC and TPR at the threshold that gives `fpr` on the normals, per anomaly set."""
    norm = df[df.label == 0]
    rows = []
    for name, g in df[df.label == 1].groupby("set"):
        both = pd.concat([norm, g])
        for s in scores:
            tau = norm[s].quantile(1 - fpr)
            rows.append({"set": name, "score": s, "auroc": auroc(both.label, both[s]),
                         "tpr": (g[s] > tau).mean()})
    return pd.DataFrame(rows).set_index(["set", "score"])


def plot_hist(df, scores=SCORES):
    fig, axes = plt.subplots(1, len(scores), figsize=(3.2 * len(scores), 2.8))
    for ax, s in zip(axes, scores):
        for name, g in df.groupby("set"):
            ax.hist(g[s], bins=40, alpha=0.5, density=True, label=name)
        ax.set_title(s)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


# ----------------------------------- main -----------------------------------
if __name__ == "__main__":
    model = torch.load(MODEL, weights_only=False, map_location=device).eval()
    scorer = LatentScorer(encode_decode(model, load(FIT))[0])

    df = score_sets(model, scorer, SETS)
    print(f"Model: {MODEL.name} \nfit: {FIT.name} \nsets: {[p.name for p, _ in SETS.values()]}")
    print(df.groupby("set")[SCORES].agg(["mean", "std"]).round(4))
    print("\n", summarize(df).round(3))

    out = BASE_DIR / "../../tmp"

    imgs = {name: load(path) for name, (path, _) in SETS.items()}
    plot_dataset_samples(model, imgs, beta=0.03, seed=10).savefig(out / "dataset_samples.png", dpi=150)
    
    df.to_csv(out / "anomaly_scores.csv", index=False)
    plot_hist(df).savefig(out / "anomaly_hist.png", dpi=150)
    plt.show()