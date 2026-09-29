"""boundary conditions and sources of one benchmark sample, stored per shape.

The domain spans R_1 x R_2 pixels with x_1 the long side. Edges are ordered left, right,
bottom, top; each carries its pixel nodes along increasing coordinate and two components
(scalar physics use the first). Every boundary node belongs to exactly one edge: traversing
the boundary counterclockwise, an edge holds its last corner but not its first (left owns
the bottom-left corner, bottom the bottom-right, right the top-right, top the top-left). Left
and right thus hold R_2 nodes, padded to R_1 and never set beyond; bottom and top hold R_1.
Sources are plain dicts, so the files load without this module.
"""

from dataclasses import asdict, dataclass

import torch

EDGES = ["left", "right", "bottom", "top"]
OFFSET = [0, 1, 1, 0]  # entry k of an edge is its node k + OFFSET along the edge
# the corner each edge passes on: (the edge owning it, its entry there, -1 the last)
PASSED = {"left": ("top", 0), "right": ("bottom", -1), "bottom": ("left", 0), "top": ("right", -1)}


def shape_name(res, aspect):
    """file tag of a domain: 256 for a square, 256x64 for aspect ratio 4."""
    return f"{res}" if aspect == 1 else f"{res}x{res // aspect}"


@dataclass
class BoundaryConditions:
    dirichlet: torch.Tensor  # (4, R_1, 2) bool
    neumann: torch.Tensor  # (4, R_1, 2) bool
    u: torch.Tensor  # (4, R_1, 2) float, prescribed values
    t: torch.Tensor  # (4, R_1, 2) float, prescribed fluxes or tractions
    shape: tuple = None  # (R_1, R_2) pixels; None in files written before rectangles, square

    @classmethod
    def empty(cls, resolution, aspect=1):
        """traction-free (homogeneous Neumann) everywhere."""
        size = (len(EDGES), resolution, 2)
        return cls(
            torch.zeros(size, dtype=torch.bool),
            torch.zeros(size, dtype=torch.bool),
            torch.zeros(size),
            torch.zeros(size),
            (resolution, resolution // aspect),
        )

    def __post_init__(self):
        if self.shape is None:
            self.shape = (self.u.shape[1],) * 2

    def edge_nodes(self, edge):
        """nodes an edge holds, one per pixel: R_2 for left and right, R_1 for bottom and top."""
        return self.shape[1] if EDGES.index(edge) < 2 else self.shape[0]

    def fix(self, edge, component, value=0.0, nodes=None, closed=False):
        """prescribe a value on the entries `nodes` (default all); `closed` also fixes the
        corner the edge passes on, through the edge that owns it."""
        e = EDGES.index(edge)
        nodes = slice(self.edge_nodes(edge)) if nodes is None else nodes
        self.dirichlet[e, nodes, component] = True
        self.neumann[e, nodes, component] = False
        self.u[e, nodes, component] = value
        if closed:
            owner, entry = PASSED[edge]
            entry = entry % self.edge_nodes(owner)
            self.fix(owner, component, value, [entry])

    def load(self, edge, component, value, nodes=None):
        e = EDGES.index(edge)
        nodes = slice(self.edge_nodes(edge)) if nodes is None else nodes
        self.neumann[e, nodes, component] = True
        self.t[e, nodes, component] = value


def source(
    kind,
    position=(0.5, 0.5),
    width=0.0,
    amplitude=(1.0, 0.0),
    frequency=None,
    cycles=None,
):
    """one source: kind "point", "gaussian" or "gravity" (position and width unused).

    frequency and cycles make it a sine burst in time (transient physics only).
    """
    return dict(
        kind=kind,
        position=tuple(position),
        width=width,
        amplitude=tuple(amplitude),
        frequency=frequency,
        cycles=cycles,
    )


def save(path, bc=None, sources=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"bc": None if bc is None else asdict(bc), "sources": list(sources)}, path
    )


def load(path):
    """(BoundaryConditions or None, list of source dicts)."""
    data = torch.load(path, weights_only=False)
    bc = None if data["bc"] is None else BoundaryConditions(**data["bc"])
    return bc, data["sources"]
