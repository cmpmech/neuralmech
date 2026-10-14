# compute reconstructions from a trained fiber VAE and MSE
# python projects/15_anomaly/fiber_vae_reconstruct.py
#
# Building blocks (also used by fiber_vae_anomaly.py):
#   out = reconstruct(model, x)                   raw CPU images (N,1,H,W) -> dict of CPU tensors
#   m   = sample_metrics(out, beta)               per-sample mse / dice / kl / anomaly score
#   plot_row(axes_row, out, m, i, label)          draw sample i on one row of axes
#   plot_dataset_samples(model, {name: images})   one random sample per dataset, one row each

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True


# ------------------------------ model and data ------------------------------
def load_model(path):
    return torch.load(path, weights_only=False, map_location=device).eval()


def load_images(path):
    """(N,H,W) .npy -> float32 CPU tensor (N,1,H,W)."""
    return torch.from_numpy(np.load(path)).to(torch.float32).unsqueeze(1)


@torch.no_grad()
def reconstruct(model, x, deterministic=False):
    """Run the VAE on raw CPU images x (N,1,H,W); everything returned is on the CPU.

    deterministic=True decodes the latent mean (what fiber_vae_anomaly.py scores);
    False uses whatever model(x) returns, as the original script did.
    'recon' and 'orig' are in raw image units; 'err' is the per-sample MSE in the
    model's (standardized) space, which is what the anomaly score 'as' has always used.
    """
    std = model.standardizer
    z = std(x)
    dev = next(model.parameters()).device
    x_pred, mean, logvar = model(z.to(dev))
    if deterministic:
        x_pred = model.decode(mean)
    x_pred, mean, logvar = x_pred.cpu(), mean.cpu(), logvar.cpu()
    return {
        "orig": x,
        "recon": std.inverse(x_pred),
        "mean": mean,
        "logvar": logvar,
        "err": (x_pred - z).pow(2).flatten(1).mean(1),
    }


# --------------------------------- metrics ----------------------------------
def dice_loss(x_pred, x, eps=1.0):
    """Soft Dice loss, reduced over the last two (H, W) dims."""
    inter = (x_pred * x).sum((-2, -1))
    denom = x_pred.sum((-2, -1)) + x.sum((-2, -1))
    return 1 - (2 * inter + eps) / (denom + eps)


def sample_metrics(out, beta=1.0):
    """Per-sample metrics from reconstruct() output; every value has leading dim N.

    'as' (anomaly score) = model-space reconstruction MSE + beta * total KL.
    'kl_dim' is the KL per latent dimension, shape (N, latent_dim).
    """
    mean, logvar = out["mean"], out["logvar"]
    kl_dim = 0.5 * (logvar.exp() + mean**2 - logvar - 1)
    kl = kl_dim.sum(1)
    return {
        "mse": (out["recon"] - out["orig"]).pow(2).flatten(1).mean(1),
        "bce": F.binary_cross_entropy(out["recon"], out["orig"], reduction="none").flatten(1).mean(1),
        "dice": dice_loss(out["recon"], out["orig"]).flatten(1).mean(1),
        "kl": kl,
        "as": out["err"] + beta * kl,
        "kl_dim": kl_dim,
    }


# --------------------------------- plotting ---------------------------------
def plot_row(axes, out, m, i=0, label=""):
    """Draw sample i on one row of axes: original | reconstruction | KL per latent dim.

    Pass only 2 axes to skip the KL panel. `label` is prepended to the first title
    (e.g. the dataset name).
    """
    left = f"mse={m['mse'][i]:.3f} | dice={m['dice'][i]:.3f} | bce={m['bce'][i]:.3f}"
    if label:
        left = f"{label}\n{left}"
    for ax, img, title in zip(axes[:2], (out["orig"][i, 0], out["recon"][i, 0]),
                              (left, f"as={m['as'][i]:.2f}")):
        ax.imshow(img, cmap="binary", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    if len(axes) > 2:
        kl_sorted = m["kl_dim"][i].sort(descending=True).values
        axes[2].bar(range(len(kl_sorted)), kl_sorted.numpy())
        axes[2].axhline(1.0, color="r", linestyle="--", linewidth=0.8, label="prior = 1")
        axes[2].set_title(f"KL/dim (total={m['kl'][i]:.2f})")
        axes[2].set_xlabel("latent dim (sorted)")
        axes[2].legend(fontsize=7)


def print_top_kl(out, m, i, k=5):
    """Print the k latent dimensions with the largest KL for sample i."""
    kl_sorted, inds = m["kl_dim"][i].sort(descending=True)
    print("\nTop latent dimensions for KL divergence:")
    for j in range(k):
        d = inds[j].item()
        print(f"dim {d}: KL={kl_sorted[j].item():.3f}, "
              f"mean={out['mean'][i, d].item():.3f}, logvar={out['logvar'][i, d].item():.3f}")


def plot_dataset_samples(model, images, beta=1.0, seed=None, kl_panel=False):
    """One random sample per dataset (one row each): original | reconstruction [| KL/dim].

    images: {name: raw CPU tensor (N,1,H,W)}. Returns the figure.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    ncol = 3 if kl_panel else 2
    fig, axes = plt.subplots(len(images), ncol, figsize=(3 * ncol, 3.3 * len(images)),
                             dpi=128, squeeze=False)
    for row, (name, x) in zip(axes, images.items()):
        i = torch.randint(len(x), (1,), generator=g)
        out = reconstruct(model, x[i], deterministic=True)
        plot_row(row, out, sample_metrics(out, beta), label=name)
    fig.tight_layout(pad=1)
    return fig


# ---------------------------------- driver ----------------------------------
def evaluate_fiber_vae(
    model_path: Path,
    data_path: Path,
    beta: float,
    num_samples: int = 4,
    seed: int = None,
) -> dict:
    """Load a saved VAE, evaluate random samples from data_path, and plot reconstructions."""
    if seed is not None:
        torch.manual_seed(seed)
    print(device)
    model = load_model(model_path)
    data = load_images(data_path)

    idx = torch.randperm(len(data))[:num_samples]
    out = reconstruct(model, data[idx])
    m = sample_metrics(out, beta)

    fig, axes = plt.subplots(num_samples, 3, figsize=(6, 3 * num_samples), dpi=128, squeeze=False)
    for row in range(num_samples):
        plot_row(axes[row], out, m, row)
        print_top_kl(out, m, row)

    fig.suptitle(f"Fiber VAE reconstruction (beta={beta})")
    fig.tight_layout(pad=1)
    plt.show()

    return {"beta": beta, "mse": m["mse"].mean().item(), "dice": m["dice"].mean().item()}


def _main():
    BASE_DIR = Path(__file__).parent
    beta = "1.0"

    # model = str(BASE_DIR / "../../models/fiber_vae_depth5_latent128_beta0.2_256.pt2")
    # model = str(BASE_DIR / f"../../models/fiber_vae_depth5_latent64_beta{beta}_256.pt2")
    # model = "../../models/bce_works.pt2"
    # model = "../../models/fiber_vae_depth5_latent40_beta1.0_256.pt2"
    # model = "../../models/bce_dice__latent40_beta1.0.pt2"
    # model = "models/bce_dice__latent40_betaBoth0.04_256.pt2"                  # "mse": 0.00320, "dice": 0.073}
    # model = "models/_fullyTrained_plateausched__latent40_beta0.08_256.pt2"  # "mse": 0.00573, "dice": 0.118} as: 9
    model = (BASE_DIR / "../../models/_latent20_beta0.03_256.pt2").resolve()                             # "mse": 0.00452, "dice": 0.111} as: 8.8

    data = str(BASE_DIR / f"../../data/t_1sq_2tot_800.npy")
    # data = (BASE_DIR / f"../../data/t_0sq_2tot_800.npy").resolve()

    samples = 4
    seed = 56

    result = evaluate_fiber_vae(Path(model), Path(data), beta=float(beta), num_samples=samples, seed=seed)
    print(json.dumps(result))


if __name__ == "__main__":
    _main()
