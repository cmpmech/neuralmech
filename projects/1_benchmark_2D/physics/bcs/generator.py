import argparse
import tomllib
from pathlib import Path

import numpy as np

from boundary import EDGES, BoundaryConditions, save, shape_name, source

BASE_DIR = Path(__file__).parent
SETUP_DIR = (BASE_DIR / "../../../../data/2D_benchmark/setups").resolve()

with open(BASE_DIR / "../../settings.toml", "rb") as f:
    SETTINGS = tomllib.load(f)["boundary_conditions"]

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=SETTINGS["count"])
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--aspect", type=int, default=1, help="long : short side")
parser.add_argument("--nfields", type=int, default=2, choices=[1, 2])
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# squares keep the plain seed, so their setups stay the same when aspect ratios are added
rng = np.random.default_rng(args.seed if args.aspect == 1 else [args.seed, args.aspect])


# ---------------------------------------- helper -------------------------------------
def segment(R):
    """entries of a full edge, a partial edge or a single node, out of the R an edge holds."""
    kind = rng.choice(["full", "partial", "node"])
    if kind == "full":
        return slice(R)
    if kind == "node":
        return [int(rng.integers(R))]
    start, stop = np.sort(rng.choice(R, size=2, replace=False))
    return slice(start, stop + 1)


def admissible(bc, nfields):
    """the fixed components suppress all rigid body modes (full rank)."""
    rows = []
    R1, R2 = bc.shape
    x = np.arange(R1 + 1) / R1  # the long side has unit length
    y = np.arange(R2 + 1) / R1
    # the nodes each edge holds, see OFFSET
    coordinates = [(0 * y[:-1], y[:-1]), (0 * y[1:] + 1, y[1:]), (x[1:], 0 * x[1:]), (x[:-1], 0 * x[:-1] + R2 / R1)]
    for edge, (x, y) in enumerate(coordinates):
        for c in range(nfields):
            for node in np.flatnonzero(bc.dirichlet[edge, :, c].numpy()):
                if nfields == 1:
                    rows.append([1.0])
                else:  # translation x, translation y, rotation
                    rows.append([c == 0, c == 1, -y[node] if c == 0 else x[node]])
    modes = 1 if nfields == 1 else 3
    return len(rows) > 0 and np.linalg.matrix_rank(np.array(rows, dtype=float)) == modes


def random_bc(R, aspect, nfields):
    while True:
        bc = BoundaryConditions.empty(R, aspect)
        count = rng.integers(1, SETTINGS["max_dirichlet_edges"] + 1)
        dirichlet = rng.choice(len(EDGES), size=count, replace=False)
        for e, edge in enumerate(EDGES):
            nodes = segment(bc.edge_nodes(edge))
            components = [c for c in range(nfields) if rng.random() < 0.5] or [
                rng.integers(nfields)
            ]
            if e in dirichlet:
                for c in components:
                    value = (
                        0.0
                        if rng.random() < SETTINGS["zero_dirichlet"]
                        else SETTINGS["value_scale"] * rng.normal()
                    )
                    bc.fix(edge, c, value, nodes)
            elif rng.random() < SETTINGS["load"]:
                for c in components:
                    bc.load(edge, c, SETTINGS["value_scale"] * rng.normal(), nodes)
        if admissible(bc, nfields):
            return bc


def random_sources(nfields):
    sources = []
    for _ in range(rng.integers(0, SETTINGS["max_sources"] + 1)):
        kind = rng.choice(["point", "gaussian", "gravity"])
        amplitude = SETTINGS["value_scale"] * rng.normal(size=nfields)
        position = rng.uniform(*SETTINGS["source_position"], size=2)
        sources.append(source(kind, position, rng.uniform(*SETTINGS["source_width"]), amplitude))
    return sources


# ---------------------------------------- export -------------------------------------
for i in range(args.count):
    name = f"random{args.nfields}_{args.seed}_{i}_{shape_name(args.resolution, args.aspect)}.pt"
    save(
        SETUP_DIR / name,
        random_bc(args.resolution, args.aspect, args.nfields),
        random_sources(args.nfields),
    )
print(f"{args.count} setups written to {SETUP_DIR}")
