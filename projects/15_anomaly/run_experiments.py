"""Run a small grid of quick experiments by launching the existing train script
with environment overrides. Saves a CSV of results and prints a summary.

This intentionally keeps code changes to the train script minimal by using
env variables to override hyperparameters.
"""
from pathlib import Path
import subprocess
import sys
import os
import json
import csv

BASE_DIR = Path(__file__).parent
TRAIN_SCRIPT = BASE_DIR / "fiber_vae_train.py"
RECON_SCRIPT = BASE_DIR / "fiber_vae_reconstruct.py"
RESULTS_CSV = BASE_DIR / "experiment_results.csv"

# quick experiment settings (keep dataset and architecture the same)
# We'll run ~10 runs selected from the parameter grid
grid = [
    {"LR": 1e-2, "WEIGHT_DECAY": 1e-4, "BETA": 0.05, "FREE_BITS": 0.0},
    {"LR": 1e-2, "WEIGHT_DECAY": 1e-4, "BETA": 0.2, "FREE_BITS": 5.0},
    {"LR": 5e-3, "WEIGHT_DECAY": 1e-4, "BETA": 0.05, "FREE_BITS": 5.0},
    {"LR": 5e-3, "WEIGHT_DECAY": 1e-3, "BETA": 0.2, "FREE_BITS": 0.0},
    {"LR": 2e-2, "WEIGHT_DECAY": 1e-4, "BETA": 0.1, "FREE_BITS": 2.0},
    {"LR": 1e-3, "WEIGHT_DECAY": 1e-4, "BETA": 0.05, "FREE_BITS": 5.0},
    {"LR": 1e-3, "WEIGHT_DECAY": 1e-3, "BETA": 0.2, "FREE_BITS": 5.0},
    {"LR": 1e-2, "WEIGHT_DECAY": 5e-4, "BETA": 0.05, "FREE_BITS": 1.0},
    {"LR": 5e-3, "WEIGHT_DECAY": 5e-4, "BETA": 0.13, "FREE_BITS": 5.0},
    {"LR": 2e-2, "WEIGHT_DECAY": 1e-5, "BETA": 0.2, "FREE_BITS": 10.0},
]

# runtime overrides to speed up experiments
EPOCHS = 8
BATCH_SIZE = 64
LATENT_DIM = 32
DEPTH = 5
DOMAIN_SIZE = 256

results = []

for i, params in enumerate(grid, 1):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"  # avoid interactive backends
    env["EPOCHS"] = str(EPOCHS)
    env["BATCH_SIZE"] = str(BATCH_SIZE)
    env["LATENT_DIM"] = str(LATENT_DIM)
    env["LR"] = str(params["LR"])
    env["WEIGHT_DECAY"] = str(params["WEIGHT_DECAY"])
    env["BETA"] = str(params["BETA"])
    env["FREE_BITS"] = str(params["FREE_BITS"])

    print(f"\n=== Run {i}/{len(grid)}: LR={env['LR']} WD={env['WEIGHT_DECAY']} BETA={env['BETA']} FREE_BITS={env['FREE_BITS']} ===")

    # run training
    try:
        subprocess.run([sys.executable, str(TRAIN_SCRIPT)], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print("Train failed:", e)
        continue

    # model filename convention used by train script
    model_file = BASE_DIR / f"../../models/fiber_vae_depth{DEPTH}_latent{LATENT_DIM}_beta{params['BETA']}_{DOMAIN_SIZE}.pt2"

    # run reconstruction eval to get mse
    try:
        cp = subprocess.run(
            [sys.executable, str(RECON_SCRIPT), "--model", str(model_file), "--beta", str(params["BETA"]), "--samples", "4", "--seed", "42"],
            check=True,
            capture_output=True,
            text=True,
        )
        # reconstruct prints JSON on stdout
        out = cp.stdout.strip().splitlines()[-1]
        res = json.loads(out)
        res.update({"LR": params["LR"], "WEIGHT_DECAY": params["WEIGHT_DECAY"], "FREE_BITS": params["FREE_BITS"]})
        results.append(res)
        print("Result:", res)
    except subprocess.CalledProcessError as e:
        print("Eval failed:", e)
        print(e.stdout)
        print(e.stderr)

# write CSV
with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["beta", "mse", "LR", "WEIGHT_DECAY", "FREE_BITS"])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

print("\nDone. Results saved to:", RESULTS_CSV)
print(results)
