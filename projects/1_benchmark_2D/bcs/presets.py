import tomllib
from pathlib import Path

from boundary import BoundaryConditions, save, shape_name, source

BASE_DIR = Path(__file__).parent
SETUP_DIR = (BASE_DIR / "../../../data/2D_benchmark/setups").resolve()

with open(BASE_DIR / "../settings.toml", "rb") as f:
    SETTINGS = tomllib.load(f)

# -------------------------------------- settings -------------------------------------
RESOLUTIONS = SETTINGS["geometry"]["resolutions"]
ASPECT_RATIOS = SETTINGS["geometry"]["aspect_ratios"]

# loads refer to a unit long side; ramped loads give the final value
TRACTION = SETTINGS["presets"]["traction"]
STRETCH = SETTINGS["presets"]["stretch"]
OPENING = SETTINGS["presets"]["opening"]
BURST = SETTINGS["presets"]["burst"]
CANTILEVER = SETTINGS["presets"]["cantilever"]
HEAT_SINK = SETTINGS["presets"]["heat_sink"]


# ---------------------------------------- setup --------------------------------------
def tension(R, aspect):
    """uniaxial tension: rollers left and bottom, traction on the right edge."""
    bc = BoundaryConditions.empty(R, aspect)
    bc.fix("left", 0, closed=True)
    bc.fix("bottom", 1, closed=True)
    bc.load("right", 0, TRACTION)
    return bc


def conduction(R, aspect):
    """temperature 0 on the left, 1 on the right, insulated top and bottom."""
    bc = BoundaryConditions.empty(R, aspect)
    bc.fix("left", 0, 0.0, closed=True)
    bc.fix("right", 0, 1.0, closed=True)
    return bc


def stretch(R, aspect):
    """uniaxial tension under displacement control."""
    bc = BoundaryConditions.empty(R, aspect)
    bc.fix("left", 0, closed=True)
    bc.fix("bottom", 1, closed=True)
    bc.fix("right", 0, STRETCH, closed=True)
    return bc


def opening(R, aspect):
    """clamped bottom, top edge pulled up (and held horizontally)."""
    bc = BoundaryConditions.empty(R, aspect)
    for component in (0, 1):
        bc.fix("bottom", component, closed=True)
    bc.fix("top", 0, closed=True)
    bc.fix("top", 1, OPENING, closed=True)
    return bc


def channel(R, aspect):
    """inflow on the left (unit velocity), outflow on the right, no-slip top and bottom."""
    bc = BoundaryConditions.empty(R, aspect)
    bc.fix("left", 0, 1.0)
    bc.fix("left", 1)
    for edge in ("bottom", "top"):
        for component in (0, 1):
            bc.fix(edge, component)
    return bc


def cantilever(R, aspect):
    """clamped left edge, point load downwards at the middle of the right edge."""
    bc = BoundaryConditions.empty(R, aspect)
    for component in (0, 1):
        bc.fix("left", component, closed=True)
    return bc, [source("point", (1.0, 0.5), amplitude=(0.0, -CANTILEVER))]


def heat_sink(R, aspect):
    """uniform heat source, zero temperature on the middle of the left edge, else insulated."""
    bc = BoundaryConditions.empty(R, aspect)
    nodes = bc.edge_nodes("left")
    start = round((0.5 - 0.5 * HEAT_SINK["length"]) * nodes)
    stop = round((0.5 + 0.5 * HEAT_SINK["length"]) * nodes)
    bc.fix("left", 0, nodes=slice(start, stop + 1))
    return bc, [source("uniform", amplitude=(HEAT_SINK["source"],))]


# ---------------------------------------- export -------------------------------------
for R in RESOLUTIONS:
    for aspect in ASPECT_RATIOS:
        shape = shape_name(R, aspect)
        save(SETUP_DIR / f"tension_{shape}.pt", tension(R, aspect))
        save(SETUP_DIR / f"conduction_{shape}.pt", conduction(R, aspect))
        save(SETUP_DIR / f"stretch_{shape}.pt", stretch(R, aspect))
        save(SETUP_DIR / f"opening_{shape}.pt", opening(R, aspect))
        save(SETUP_DIR / f"channel_{shape}.pt", channel(R, aspect))
        # reflecting edges, a sine burst from a point source in the centre
        burst = source("point", BURST["position"], BURST["width"], (1.0,), BURST["frequency"], BURST["cycles"])
        save(SETUP_DIR / f"burst_{shape}.pt", BoundaryConditions.empty(R, aspect), [burst])
        save(SETUP_DIR / f"cantilever_{shape}.pt", *cantilever(R, aspect))
        save(SETUP_DIR / f"heat_sink_{shape}.pt", *heat_sink(R, aspect))
print(f"setups written to {SETUP_DIR}")
