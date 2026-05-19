from pathlib import Path

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
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples), dpi=128)
    if num_samples == 1:
        axes = axes.reshape(1, 2)
    
    standardizex = model.standardizer
    mse_list = []
    with torch.no_grad():
        x = standardizex(data)
        x_pred, mean_pred, logvar_pred = model(x)
        mse = F.mse_loss(x_pred, x, reduction="mean").item()
        
        for row, idx in enumerate(sample_indices.tolist()):
            orig = standardizex.inverse(x[idx : idx + 1])[0, 0]
            recon = standardizex.inverse(x_pred[idx : idx + 1])[0, 0]
            
            # mean and logvar averages across latent variables
            mean_val = mean_pred[idx].mean().item()
            logvar_val = logvar_pred[idx].mean().item()
            
            mse = F.mse_loss(recon, orig, reduction="mean").item()
            mse_list.append(mse)

            axes[row, 0].imshow(orig.cpu(), cmap="binary", vmin=0, vmax=1)
            axes[row, 0].set_title("original")
            axes[row, 1].imshow(recon.cpu(), cmap="binary", vmin=0, vmax=1)
            axes[row, 1].set_title(f"mean={mean_val:.3f}, logvar={logvar_val:.3f}, mse={mse:.3f}")
            for ax in axes[row]:
                ax.set_aspect("equal")
                ax.axis("off")

        fig.suptitle(f"Fiber VAE reconstruction (beta={beta})")
        fig.tight_layout(pad=1)
        plt.show()

    return {"beta": beta, "mse": np.mean(mse_list).item()}


BETA = 0.1
LATENT = 32
MODEL_PATH = BASE_DIR / f"../../models/fiber_vae_depth5_latent{LATENT}_beta{BETA}_256.pt2"
DATA_PATH = BASE_DIR / "../../data/fibers_256.npy"

result = evaluate_fiber_vae(MODEL_PATH, DATA_PATH, beta=BETA)
print(result)
