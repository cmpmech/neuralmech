from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
USE_RATIOS = False # append water/cement and coarse/fine as extra features

# admissible mixture box, kg per m^3 except age in days; the dataset is cropped to it
# and candidate mixtures are drawn inside it. the bounds are the component extremes of
# the dataset, with age cut off before the 180 day jump (971 of 1030 mixes survive)
LOWER = np.array([102.0, 0.0, 0.0, 121.0, 0.0, 801.0, 594.0, 1.0])
UPPER = np.array([540.0, 360.0, 201.0, 247.0, 33.0, 1145.0, 993.0, 120.0])

# the box alone still admits mixtures of several tonnes per cubic metre, so the total
# mass of the seven components is bounded as well (2195 to 2551 kg in the dataset)
LOWER_MASS = 2195.0
UPPER_MASS = 2551.0

LABELS = [
    "cement",
    "slag",
    "fly ash",
    "water",
    "superplasticizer",
    "coarse aggregate",
    "fine aggregate",
    "age",
]


# --------------------------------------- helper --------------------------------------
def features(X: np.ndarray) -> np.ndarray:
    """Map raw mixture components to model inputs, optionally adding the two ratios."""
    if not USE_RATIOS:
        return X
    water_cement = X[:, 3] / X[:, 0]
    coarse_fine = X[:, 5] / X[:, 6]
    return np.column_stack([X, water_cement, coarse_fine])


def sample_mixtures(rng: np.random.Generator, draws: int) -> np.ndarray:
    """Draw admissible mixtures uniformly in the box, rejecting inadmissible masses.

    Roughly 40 percent of the draws survive, so the returned array is shorter than
    `draws`.
    """
    X = rng.uniform(LOWER, UPPER, (draws, len(LOWER)))
    mass = X[:, :7].sum(axis=1)
    return X[(mass >= LOWER_MASS) & (mass <= UPPER_MASS)]


def load_concrete() -> tuple:
    """Load the dataset cropped to the admissible box.

    Returns standardized inputs, strengths in mpa, and the standardizer so that
    candidate mixtures drawn in the box can be mapped into the same space. the
    inputs must be standardized because raw age cubed is O(1e6) and would ruin the
    conditioning of the polynomial feature matrix; the strengths are left in mpa so
    that noise, uncertainty and error all read in mpa.
    """
    data = np.load(DATA_DIR / "concrete.npz")
    X, Y = data["X"], data["Y"]

    ids = np.all((X >= LOWER) & (X <= UPPER), axis=1)
    X, Y = features(X[ids]), Y[ids]

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    standardizex = lambda X: (X - X_mean) / X_std
    return standardizex(X), Y, standardizex
