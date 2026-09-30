"""one bilinear element per pixel, and the dofs the benchmark setups act on."""

import mlhp
import numpy as np

from boundary import OFFSET


def lengths(shape):
    """side lengths of a domain of `shape` pixels, the long side x_1 of unit length."""
    return [1.0, shape[1] / shape[0]]


class PixelMesh:
    """one bilinear element per square pixel, so the dofs sit on the pixel nodes.

    `dof[i, j, c]` is the dof of component c at node (x_i, y_j).
    """

    def __init__(self, resolution, nfields, aspect=1):
        self.R, self.nfields = resolution, nfields
        self.shape = (resolution, resolution // aspect)
        self.grid = mlhp.makeRefinedGrid(mlhp.makeGrid(list(self.shape), lengths(self.shape)))
        self.basis = mlhp.makeHpTensorSpace(self.grid, degree=1, nfields=nfields)
        self.ndof = self.basis.ndof()

        # linear functions are reproduced exactly, so projecting x and y yields the nodes
        ij = []
        for coordinate in ("x", "y"):
            field = mlhp.scalarField(2, coordinate)
            if nfields > 1:
                field = mlhp.vectorField(2, [field] * nfields)
            ij.append(
                np.rint(np.array(mlhp.projectOnto(self.basis, field)) * resolution)
            )
        component = np.array(self.basis.fieldmap()) if nfields > 1 else 0
        self.dof = np.zeros((self.shape[0] + 1, self.shape[1] + 1, nfields), dtype=int)
        self.dof[ij[0].astype(int), ij[1].astype(int), component] = np.arange(self.ndof)

    def closed_edge_dofs(self, edge):
        """dofs (nodes, nfields) of all nodes along edge 0-3 (left, right, bottom, top)."""
        return [self.dof[0], self.dof[-1], self.dof[:, 0], self.dof[:, -1]][edge]

    def edge_dofs(self, edge):
        """dofs of the nodes edge 0-3 holds: all but the corner it passes on."""
        dofs = self.closed_edge_dofs(edge)
        return dofs[1:] if OFFSET[edge] else dofs[:-1]

    def dirichlet(self, bc, scale=1.0):
        """mlhp dirichlet dofs [indices, values], prescribed values scaled by `scale`."""
        ids, values = [], []
        for edge in range(4):
            dofs = self.edge_dofs(edge)
            mask = bc.dirichlet[edge, : len(dofs), : self.nfields].numpy()
            ids.append(dofs[mask])
            values.append(scale * bc.u[edge, : len(dofs), : self.nfields].numpy()[mask])
        ids, first = np.unique(np.concatenate(ids), return_index=True)
        return [ids.tolist(), np.concatenate(values)[first].tolist()]

    def neumann(self, bc, scale=1.0):
        """nodal loads from the edge tractions, lumped with the trapezoidal rule."""
        h = 1.0 / self.R
        load = np.zeros(self.ndof)
        for edge in range(4):
            dofs = self.closed_edge_dofs(edge)
            weights = np.full(len(dofs), h)
            weights[[0, -1]] = 0.5 * h
            t = (bc.t * bc.neumann)[edge, : len(dofs) - 1, : self.nfields].numpy()
            # the traction reaches the corner the edge passes on, continued from its neighbor
            t = np.concatenate([t[:1], t]) if OFFSET[edge] else np.concatenate([t, t[-1:]])
            np.add.at(load, dofs.ravel(), (weights[:, None] * t).ravel())
        return scale * load

    def sources(self, sources, g, scale=1.0):
        """nodal loads of point, gaussian, uniform and gravity sources (nodal quadrature).

        Positions are fractions of each side. Gravity scales with the phase g averaged onto
        the nodes (density ~ gray value), a uniform source does not.
        """
        (R1, R2), h = self.shape, 1.0 / self.R
        x, y = np.meshgrid(np.arange(R1 + 1) * h, np.arange(R2 + 1) * h, indexing="ij")
        area = np.full((R1 + 1, R2 + 1), h * h)
        area[[0, -1]] *= 0.5
        area[:, [0, -1]] *= 0.5
        load = np.zeros(self.ndof)
        for s in sources:
            amplitude = np.array(s["amplitude"])[: self.nfields]
            if s["kind"] == "point":
                i, j = (int(round(p * n)) for p, n in zip(s["position"], self.shape))
                load[self.dof[i, j]] += amplitude
                continue
            if s["kind"] == "gaussian":
                w = s["width"]
                x0, y0 = (p * n * h for p, n in zip(s["position"], self.shape))
                r2 = (x - x0) ** 2 + (y - y0) ** 2
                density = np.exp(-0.5 * r2 / w**2) / (2 * np.pi * w**2)
            elif s["kind"] == "uniform":
                density = np.ones_like(x)
            else:
                padded = np.pad(g, 1, mode="edge")
                density = 0.25 * (
                    padded[1:, 1:]
                    + padded[:-1, 1:]
                    + padded[1:, :-1]
                    + padded[:-1, :-1]
                )
            np.add.at(
                load,
                self.dof.reshape(-1, self.nfields),
                (density * area).reshape(-1, 1) * amplitude,
            )
        return scale * load

    def cell_values(self, function):
        """(R_1, R_2, ncomponents) values of an mlhp mesh function at the pixel centres."""
        ncells = self.shape[0] * self.shape[1]
        values = np.array([function(cell, [0.0, 0.0]) for cell in range(ncells)])
        if not hasattr(self, "pixel"):
            xy = mlhp.vectorField(
                2, [mlhp.scalarField(2, "x"), mlhp.scalarField(2, "y")]
            )
            centres = mlhp.meshFunction(self.grid, xy)
            ij = np.array([centres(cell, [0.0, 0.0]) for cell in range(ncells)])
            self.pixel = tuple(np.floor(ij * self.R).astype(int).T)
        out = np.zeros((*self.shape, values.shape[1]))
        out[self.pixel] = values
        return out

    @staticmethod
    def pixel_mean(nodal):
        """(R_1, R_2, ...) pixel averages of nodal values (R_1 + 1, R_2 + 1, ...)."""
        return 0.25 * (
            nodal[1:, 1:] + nodal[:-1, 1:] + nodal[1:, :-1] + nodal[:-1, :-1]
        )

    def internal(self, load, dirichlet):
        """restrict a full-length vector to the free dofs (mlhp's reduced ordering)."""
        return np.delete(load, dirichlet[0])

    def nodal(self, dofs):
        """(R_1 + 1, R_2 + 1, nfields) nodal values of a full dof vector."""
        return np.asarray(dofs)[self.dof]
