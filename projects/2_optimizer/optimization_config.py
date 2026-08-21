import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()


@dataclass(frozen=True)
class Problem:
    f: Callable
    guess: list
    xrange: tuple
    yrange: tuple


# --------------------------------- objective functions -------------------------------
rosenbrock = Problem(
    f=lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
    guess=[3.0, 3.0],
    xrange=(-2, 4),
    yrange=(-1, 5),
)

ackley = Problem(
    f=lambda x: (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
        + 20
        + np.exp(1)
    ),
    guess=[3, 3.2],
    xrange=(-4, 4),
    yrange=(-4, 4),
)


# ----------------------------------- postprocessing ----------------------------------
def plot(problem: Problem, name: str, book: bool, resolution: int = 800):
    x = np.linspace(*problem.xrange, resolution)
    y = np.linspace(*problem.yrange, resolution)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = problem.f(np.stack([xx, yy], axis=0))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=resolution // 4)
    ax.contourf(xx, yy, z, levels=48, cmap="cividis")
    ax.plot(0, 0, "ws", ms=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_rasterized(True)  # avoid contourline artifacts
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if book:
        fig.savefig(RGB_PDF_DIR / f"{name}.pdf")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", action="store_true")
    args = parser.parse_args()

    plot(ackley, "ackley", args.book)
