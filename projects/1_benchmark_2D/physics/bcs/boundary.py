"""boundary conditions and sources of one benchmark sample, stored per resolution.

Edges are ordered left, right, bottom, top; each carries the R + 1 pixel nodes along
increasing coordinate and two components (scalar physics use the first). Sources are
plain dicts, so the files load without this module.
"""

from dataclasses import asdict, dataclass

import torch

EDGES = ["left", "right", "bottom", "top"]


@dataclass
class BoundaryConditions:
    dirichlet: torch.Tensor  # (4, R + 1, 2) bool
    neumann: torch.Tensor  # (4, R + 1, 2) bool
    u: torch.Tensor  # (4, R + 1, 2) float, prescribed values
    t: torch.Tensor  # (4, R + 1, 2) float, prescribed fluxes or tractions

    @classmethod
    def empty(cls, resolution):
        """traction-free (homogeneous Neumann) everywhere."""
        shape = (len(EDGES), resolution + 1, 2)
        return cls(
            torch.zeros(shape, dtype=torch.bool),
            torch.zeros(shape, dtype=torch.bool),
            torch.zeros(shape),
            torch.zeros(shape),
        )

    def fix(self, edge, component, value=0.0, nodes=slice(None)):
        e = EDGES.index(edge)
        self.dirichlet[e, nodes, component] = True
        self.neumann[e, nodes, component] = False
        self.u[e, nodes, component] = value

    def load(self, edge, component, value, nodes=slice(None)):
        e = EDGES.index(edge)
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
