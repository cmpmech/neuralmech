from pathlib import Path

from boundary import BoundaryConditions, save, source

BASE_DIR = Path(__file__).parent
SETUP_DIR = (BASE_DIR / "../../../../data/2D_benchmark/setups").resolve()

# -------------------------------------- settings -------------------------------------
RESOLUTIONS = [128, 256]

# loads refer to the unit square; ramped loads give the final value
TRACTION = 1.0  # elasticity, right edge
STRETCH = 2e-3  # plasticity, right edge displacement as a fraction of the width
OPENING = 2e-2  # fracture, top edge displacement


# ---------------------------------------- setup --------------------------------------
def tension(R):
    """uniaxial tension: rollers left and bottom, traction on the right edge."""
    bc = BoundaryConditions.empty(R)
    bc.fix("left", 0)
    bc.fix("bottom", 1)
    bc.load("right", 0, TRACTION)
    return bc


def conduction(R):
    """temperature 0 on the left, 1 on the right, insulated top and bottom."""
    bc = BoundaryConditions.empty(R)
    bc.fix("left", 0, 0.0)
    bc.fix("right", 0, 1.0)
    return bc


def stretch(R):
    """uniaxial tension under displacement control."""
    bc = BoundaryConditions.empty(R)
    bc.fix("left", 0)
    bc.fix("bottom", 1)
    bc.fix("right", 0, STRETCH)
    return bc


def opening(R):
    """clamped bottom, top edge pulled up (and held horizontally)."""
    bc = BoundaryConditions.empty(R)
    for component in (0, 1):
        bc.fix("bottom", component)
    bc.fix("top", 0)
    bc.fix("top", 1, OPENING)
    return bc


def channel(R):
    """inflow on the left (unit velocity), outflow on the right, no-slip top and bottom."""
    bc = BoundaryConditions.empty(R)
    bc.fix("left", 0, 1.0)
    bc.fix("left", 1)
    for edge in ("bottom", "top"):
        for component in (0, 1):
            bc.fix(edge, component)
    return bc


# ---------------------------------------- export -------------------------------------
for R in RESOLUTIONS:
    save(SETUP_DIR / f"tension_{R}.pt", tension(R))
    save(SETUP_DIR / f"conduction_{R}.pt", conduction(R))
    save(SETUP_DIR / f"stretch_{R}.pt", stretch(R))
    save(SETUP_DIR / f"opening_{R}.pt", opening(R))
    save(SETUP_DIR / f"channel_{R}.pt", channel(R))
    # reflecting edges, a sine burst from a point source in the centre
    burst = source("point", (0.5, 0.5), 0.02, (1.0,), frequency=30.0, cycles=2)
    save(SETUP_DIR / f"burst_{R}.pt", BoundaryConditions.empty(R), [burst])
print(f"setups written to {SETUP_DIR}")
