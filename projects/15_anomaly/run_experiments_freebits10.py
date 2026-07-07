"""Run a small grid of experiments with FREE_BITS pinned at 10 and EPOCHS=44.
Saves train/val MSE for each run to `experiments_freebits10_results.csv`.

Usage: run directly with the repo Python (we use `sys.executable`).
"""
from pathlib import Path
import subprocess
import sys
import os
import csv
import json

import torch
import torch.nn.functional as F
import numpy as np

BASE_DIR = Path(__file__).parent
TRAIN_SCRIPT = BASE_DIR / "fiber_vae_train.py"
RESULTS_CSV = BASE_DIR / "experiments_freebits10_results.csv"

# Hand-picked 10 experiments (LR, WEIGHT_DECAY, BETA)
experiments = [
    {"LR": 0.01, "WEIGHT_DECAY": 1e-4, "BETA": 0.2},
    {"LR": 0.02, "WEIGHT_DECAY": 1e-4, "BETA": 0.2},
    {"LR": 0.005, "WEIGHT_DECAY": 1e-4, "BETA": 0.2},
    {"LR": 0.01, "WEIGHT_DECAY": 1e-5, "BETA": 0.2},
    {"LR": 0.01, "WEIGHT_DECAY": 1e-6, "BETA": 0.2},
    {"LR": 0.01, "WEIGHT_DECAY": 1e-4, "BETA": 0.1},
    {"LR": 0.01, "WEIGHT_DECAY": 1e-4, "BETA": 0.05},
    {"LR": 0.02, "WEIGHT_DECAY": 1e-5, "BETA": 0.1},
    {"LR": 0.005, "WEIGHT_DECAY": 1e-5, "BETA": 0.1},
    {"LR": 0.02, "WEIGHT_DECAY": 1e-4, "BETA": 0.05},
]

# Fixed settings
FREE_BITS = 10.0
EPOCHS = 44
BATCH_SIZE = 64
LATENT_DIM = 32
DEPTH = 5
DOMAIN_SIZE = 256

results = []

for exp in experiments:
    lr = exp["LR"]
    wd = exp["WEIGHT_DECAY"]
    beta = exp["BETA"]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["EPOCHS"] = str(EPOCHS)
    env["BATCH_SIZE"] = str(BATCH_SIZE)
    env["LATENT_DIM"] = str(LATENT_DIM)
    env["LR"] = str(lr)
    env["WEIGHT_DECAY"] = str(wd)
    env["BETA"] = str(beta)
    env["FREE_BITS"] = str(FREE_BITS)

    print(f"\n--- Running LR={lr} WD={wd} BETA={beta} (FREE_BITS={FREE_BITS}) ---")
    try:
        subprocess.run([sys.executable, str(TRAIN_SCRIPT)], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print("Training failed for experiment:", exp, e)
        continue

    # load saved model (train script uses the same naming pattern)
    model_file = BASE_DIR / f"../../models/fiber_vae_depth{DEPTH}_latent{LATENT_DIM}_beta{beta}_{DOMAIN_SIZE}.pt2"
    if not model_file.exists():
        print("Model file not found:", model_file)
        continue

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_file, weights_only=False, map_location=device)
    model.eval()

    # load data and reproduce train/val split deterministically
    data = torch.from_numpy(np.load(BASE_DIR / f"../../data/fibers_{DOMAIN_SIZE}.npy")).to(torch.float32).unsqueeze(1)

    torch.manual_seed(42)
    n = len(data)
    n_train = int(0.9 * n)
    indices = torch.randperm(n).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    standardizer = model.standardizer

    # compute train/val MSE (on standardized data as model expects)
    with torch.no_grad():
        x_train = torch.stack([data[i] for i in train_idx]).to(device)
        x_val = torch.stack([data[i] for i in val_idx]).to(device)

        x_train_s = standardizer(x_train)
        x_train_pred, _, _ = model(x_train_s.to(device))
        train_mse = F.mse_loss(x_train_pred, x_train_s, reduction="mean").item()

        x_val_s = standardizer(x_val)
        x_val_pred, _, _ = model(x_val_s.to(device))
        val_mse = F.mse_loss(x_val_pred, x_val_s, reduction="mean").item()

    row = {"LR": lr, "WEIGHT_DECAY": wd, "BETA": beta, "FREE_BITS": FREE_BITS, "train_mse": train_mse, "val_mse": val_mse}
    results.append(row)
    print(f"LR={lr} WD={wd} BETA={beta} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

# write CSV
with open(RESULTS_CSV, "w", newline="") as f:
    fieldnames = ["LR", "WEIGHT_DECAY", "BETA", "FREE_BITS", "train_mse", "val_mse"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("Experiments complete. Results saved to:", RESULTS_CSV)
print(json.dumps(results, indent=2))
