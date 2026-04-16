import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm
from optimization_config import rosenbrock as objective

torch.backends.cudnn.deterministic = True

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

f, xrange, yrange, guess = (
    objective.f,
    objective.xrange,
    objective.yrange,
    objective.guess,
)

EPOCHS = 2000


# ------------------------------ optimizer -------------------------------
def optimize(optimizer_cls, lr, use_closure=False, **kwargs):
    x = torch.nn.Parameter(torch.tensor(guess))
    optimizer = optimizer_cls([x], lr=lr, **kwargs)
    trajectory = [x.data.clone()]
    for _ in range(EPOCHS):
        if use_closure:

            def closure():
                optimizer.zero_grad()
                loss = f(x)
                loss.backward()
                return loss

            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            f(x).backward()
            optimizer.step()
        trajectory.append(x.data.clone())
    return torch.stack(trajectory)


# ---------------------- optimization trajectories -----------------------
trajectories = {
    "steepest": optimize(torch.optim.SGD, 1e-4),
    "momentum": optimize(torch.optim.SGD, 1e-4, momentum=0.8),
    "adagrad": optimize(torch.optim.Adagrad, 1),
    "rmsprop": optimize(torch.optim.RMSprop, 5e-2),
    "adam": optimize(torch.optim.Adam, 0.8),
    "lbfgs": optimize(torch.optim.LBFGS, 0.1, use_closure=True, max_iter=20),
}


# ---------------------------- postprocessing ----------------------------
STYLES = [
    ("rmsprop", "w", "-", {"alpha": 0.6}),
    ("lbfgs", "c", "-", {"marker": "o"}),
    ("adam", "k", ":", {}),
    ("adagrad", "purple", "--", {}),
    ("momentum", "r", "--", {}),
    ("steepest", "b", ":", {}),
]


def plot(xr, yr, filename, final_markers=False):
    x = torch.linspace(*xr, 100)
    y = torch.linspace(*yr, 100)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    z = f(torch.stack([xx, yy], dim=0))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    levels = torch.logspace(torch.log10(z.min()), torch.log10(z.max()), 24)
    ax.contourf(xx, yy, z, norm=LogNorm(), levels=levels, cmap="cividis")

    for name, color, ls, extra in STYLES:
        t = trajectories[name]
        ax.plot(t[:, 0], t[:, 1], color=color, linestyle=ls, linewidth=3, **extra)

    if final_markers:
        for name, color in [("adam", "k"), ("momentum", "r")]:
            t = trajectories[name]
            ax.plot(t[-1, 0], t[-1, 1], color + "o", linewidth=3)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.axis("off")
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)

    if args.book:
        fig.savefig(RESULTS_DIR / filename)
    else:
        plt.show()
    plt.close(fig)


plot(xrange, yrange, "rosenbrock.png")
plot((0.8, 1.2), (0.8, 1.2), "rosenbrock_zoomed.png", final_markers=True)
