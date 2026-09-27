import argparse
from pathlib import Path

import numpy as np

from boundary import EDGES, BoundaryConditions, save, source

BASE_DIR = Path(__file__).parent
SETUP_DIR = (BASE_DIR / "../../../../data/2D_benchmark/setups").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=10)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--nfields", type=int, default=2, choices=[1, 2])
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# -------------------------------------- settings -------------------------------------
MAX_DIRICHLET_EDGES = 2  # clamped edges beyond this are exotic
ZERO_DIRICHLET = 0.7  # probability that a prescribed value is a plain support (zero)
LOAD = 0.5  # probability that a free edge carries a load
MAX_SOURCES = 2
VALUE_SCALE = 1.0  # magnitude of prescribed values, loads and sources


# ---------------------------------------- helper -------------------------------------
def segment(R):
    """nodes of a full edge, a partial edge or a single corner."""
    kind = rng.choice(["full", "partial", "corner"])
    if kind == "full":
        return slice(None)
    if kind == "corner":
        return [0] if rng.random() < 0.5 else [R]
    start, stop = np.sort(rng.choice(R + 1, size=2, replace=False))
    return slice(start, stop + 1)


def admissible(bc, nfields):
    """the fixed components suppress all rigid body modes (full rank)."""
    rows = []
    R = bc.dirichlet.shape[1] - 1
    s = np.linspace(0.0, 1.0, R + 1)
    coordinates = [(0 * s, s), (0 * s + 1, s), (s, 0 * s), (s, 0 * s + 1)]
    for edge, (x, y) in enumerate(coordinates):
        for c in range(nfields):
            for node in np.flatnonzero(bc.dirichlet[edge, :, c].numpy()):
                if nfields == 1:
                    rows.append([1.0])
                else:  # translation x, translation y, rotation
                    rows.append([c == 0, c == 1, -y[node] if c == 0 else x[node]])
    modes = 1 if nfields == 1 else 3
    return len(rows) > 0 and np.linalg.matrix_rank(np.array(rows, dtype=float)) == modes


def random_bc(R, nfields):
    while True:
        bc = BoundaryConditions.empty(R)
        count = rng.integers(1, MAX_DIRICHLET_EDGES + 1)
        dirichlet = rng.choice(len(EDGES), size=count, replace=False)
        for e, edge in enumerate(EDGES):
            nodes = segment(R)
            components = [c for c in range(nfields) if rng.random() < 0.5] or [
                rng.integers(nfields)
            ]
            if e in dirichlet:
                for c in components:
                    value = (
                        0.0
                        if rng.random() < ZERO_DIRICHLET
                        else VALUE_SCALE * rng.normal()
                    )
                    bc.fix(edge, c, value, nodes)
            elif rng.random() < LOAD:
                for c in components:
                    bc.load(edge, c, VALUE_SCALE * rng.normal(), nodes)
        if admissible(bc, nfields):
            return bc


def random_sources(nfields):
    sources = []
    for _ in range(rng.integers(0, MAX_SOURCES + 1)):
        kind = rng.choice(["point", "gaussian", "gravity"])
        amplitude = VALUE_SCALE * rng.normal(size=nfields)
        sources.append(
            source(
                kind, rng.uniform(0.1, 0.9, size=2), rng.uniform(0.01, 0.1), amplitude
            )
        )
    return sources


# ---------------------------------------- export -------------------------------------
for i in range(args.count):
    name = f"random{args.nfields}_{args.seed}_{i}_{args.resolution}.pt"
    save(
        SETUP_DIR / name,
        random_bc(args.resolution, args.nfields),
        random_sources(args.nfields),
    )
print(f"{args.count} setups written to {SETUP_DIR}")
