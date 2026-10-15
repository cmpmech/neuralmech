import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.utils.benchmark as benchmark
from torch import nn

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
torch.manual_seed(2)
torch.backends.cudnn.deterministic = True
WIDTHS = [32, 128, 512, 2048]  # N
MIN_RUN_TIME = 0.1  # in s
MAX_BATCH_POWER = {"cpu": 16, "cuda": 20}  # largest batch size B = 2**power

# ---------------------------------- parametric study ---------------------------------
# GPU first, so that the CPU sweep can be truncated to its shorter batch range
times = {}
for device in ["cuda", "cpu"]:
    batches = [2**i for i in range(MAX_BATCH_POWER[device])]
    for grad in [False, True]:
        for N in WIDTHS:
            model = nn.Linear(N, N).to(device)
            x = torch.randn(max(batches), N, device=device)
            timings = []
            for B in batches:
                xb = x[:B].clone().requires_grad_(grad)
                if grad:
                    stmt = "loss = model(x).sum(); loss.backward()"
                else:
                    stmt = "model(x)"
                timer = benchmark.Timer(stmt=stmt, globals={"model": model, "x": xb})
                timings.append(timer.blocked_autorange(min_run_time=MIN_RUN_TIME).mean)
            times[device, grad, N] = timings
            print(f"{device} grad={grad} N={N} done")

# --------------------------------------- export --------------------------------------
# y1: GPU, y2: CPU, on the batch range both share
if args.book:
    for grad in [False, True]:
        for N in WIDTHS:
            cpu = times["cpu", grad, N]
            save_csv(
                CSV_DIR / f"fnn_{N}_{grad}.csv",
                x=[2**i for i in range(len(cpu))],
                y1=times["cuda", grad, N][: len(cpu)],
                y2=cpu,
            )

# ----------------------------------- postprocessing ----------------------------------
fig, axes = plt.subplots(1, 2, sharey=True)
for ax, grad in zip(axes, [False, True]):
    for N in WIDTHS:
        for device, style in [("cuda", "-"), ("cpu", "--")]:
            ax.plot([2**i for i in range(len(times[device, grad, N]))],
                    times[device, grad, N], style, label=f"{device} N={N}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
axes[0].set_ylabel("wall-clock time in s")
axes[0].legend(fontsize=6)
plt.show()
