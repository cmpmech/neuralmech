# compute reconstructions from a trained fiber VAE and MSE
# python projects/15_anomaly/fiber_vae_reconstruct.py --model models/fiber_vae_depth5_latent128_beta0.3_256.pt2

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True


def evaluate_fiber_vae(
    model_path: Path, 
    data_path: Path,
    beta: float,
    num_samples: int = 4,
    seed: int = None,
) -> dict:
    """Load a saved VAE, evaluate on fibers_256, and plot reconstructions."""
    if seed is not None:
        torch.manual_seed(seed)
    print(device)
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    data = torch.from_numpy(np.load(data_path)).to(torch.float32).unsqueeze(1).to(device)
    sample_indices = torch.randperm(len(data))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(6, 3 * num_samples), dpi=128)
    if num_samples == 1:
        axes = axes.reshape(1, 3)
    
    standardizex = model.standardizer
    mse_list = []
    with torch.no_grad():
        x = standardizex(data)
        x_pred, mean_pred, logvar_pred = model(x)
        mse = F.mse_loss(x_pred, x, reduction="mean").item()
        
        for row, sample_idx in enumerate(sample_indices.tolist()):
            orig = standardizex.inverse(x[sample_idx : sample_idx + 1])[0, 0]
            recon = standardizex.inverse(x_pred[sample_idx : sample_idx + 1])[0, 0]

            # per-sample latent stats
            mean_i    = mean_pred[sample_idx]                          # (latent_dim,)
            logvar_i  = logvar_pred[sample_idx]                        # (latent_dim,)
            var_i     = torch.exp(logvar_i)

            # KL per dimension — this is the meaningful quantity
            kl_per_dim = 0.5 * (var_i + mean_i**2 - logvar_i - 1)  # (latent_dim,)

            # per-sample anomaly score
            recon_err     = F.mse_loss(x_pred[sample_idx], x[sample_idx], reduction="mean")
            kl_per_sample = kl_per_dim.sum()
            anomaly_score = (recon_err + beta * kl_per_sample).item()

            mse = F.mse_loss(recon, orig, reduction="mean").item()
            mse_list.append(mse)

            axes[row, 0].imshow(orig.cpu(), cmap="binary", vmin=0, vmax=1)
            axes[row, 0].set_title(f"mse={mse:.3f}")
            axes[row, 1].imshow(recon.cpu(), cmap="binary", vmin=0, vmax=1)
            axes[row, 1].set_title(f"as={anomaly_score:.2f}")

            # --- better plot: KL per latent dim, sorted descending ---
            kl_sorted, kl_sorted_inds = kl_per_dim.cpu().sort(descending=True)
            axes[row, 2].bar(range(len(kl_sorted)), kl_sorted.numpy())
            axes[row, 2].axhline(1.0, color="r", linestyle="--", linewidth=0.8, label="prior = 1")
            axes[row, 2].set_title(f"KL/dim (total={kl_per_sample:.2f})")
            axes[row, 2].set_xlabel("latent dim (sorted)")
            axes[row, 2].legend(fontsize=7)

            # print the 5 dimensions with maximum KL divergence
            print("\nTop latent dimensions for KL divergence:")
            for i in range(5):
                dim_idx = kl_sorted_inds[i].item()
                print(f"dim {dim_idx}: KL={kl_sorted[i].item():.3f}, mean={mean_i[i].item():.3f}, logvar={logvar_i[i]:.3f}")

            for ax in axes[row, 0:2]:
                ax.set_aspect("equal")
                ax.axis("off")

        fig.suptitle(f"Fiber VAE reconstruction (beta={beta})")
        fig.tight_layout(pad=1)
        plt.show()

    return {"beta": beta, "mse": np.mean(mse_list).item()}


def _main():
    beta = "0.2"
    # model = str(BASE_DIR / "../../models/fiber_vae_depth5_latent128_beta0.2_256.pt2")
    model = str(BASE_DIR / f"../../models/fiber_vae_depth5_latent64_beta{beta}_256.pt2")
    data = str(BASE_DIR / f"../../data/t_1sq_2tot_400.npy")
    # data = str(BASE_DIR / f"../../data/t_circ1min_3max_400.npy")

    # data = str(BASE_DIR / "../../data/fibers_anomaly_1_256_10.npy")
    # data = str(BASE_DIR / "../../data/fibers_256.npy")
    samples = 4
    seed = 56

    result = evaluate_fiber_vae(Path(model), Path(data), beta=float(beta), num_samples=samples, seed=seed)
    print(json.dumps(result))


if __name__ == "__main__":
    _main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--data", type=str, default=str(BASE_DIR / "../../data/fibers_256.npy"))
    # parser.add_argument("--beta", type=float, default=0.3)
    # parser.add_argument("--samples", type=int, default=4)
    # parser.add_argument("--seed", type=int, default=42)
    # args = parser.parse_args()

    # result = evaluate_fiber_vae(Path(args.model), Path(args.data), beta=args.beta, num_samples=args.samples, seed=args.seed)
    # print(json.dumps(result))
