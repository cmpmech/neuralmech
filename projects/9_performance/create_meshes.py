from pathlib import Path

import gmsh
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
# geometry
LENGTH = 1.0
RADIUS = 0.5  # hole centered at the origin

# discretization
ELEMENTS = [2**k for k in range(7)]  # per side of each of the two patches, per level
H_ORDERS = [1, 3]  # geometric and polynomial order of the h-refinement, on every level
P_LEVEL = 0  # level of the p-refinement, 2 quadrilaterals
P_ORDERS = range(1, 11)  # gmsh quadrilaterals end at order 10

meshes = {(order, level) for order in H_ORDERS for level in range(len(ELEMENTS))}
meshes |= {(order, P_LEVEL) for order in P_ORDERS}

# ---------------------------------------- mesh ---------------------------------------
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

for order, level in sorted(meshes):
    n = ELEMENTS[level]
    # two transfinite patches split along the diagonal, so that every cell is a quadrilateral
    gmsh.model.add("perforated_plate")
    geo = gmsh.model.geo
    diagonal = RADIUS / np.sqrt(2.0)
    center = geo.addPoint(0.0, 0.0, 0.0)
    arc_start = geo.addPoint(RADIUS, 0.0, 0.0)
    arc_mid = geo.addPoint(diagonal, diagonal, 0.0)
    arc_end = geo.addPoint(0.0, RADIUS, 0.0)
    bottom_right = geo.addPoint(LENGTH, 0.0, 0.0)
    top_right = geo.addPoint(LENGTH, LENGTH, 0.0)
    top_left = geo.addPoint(0.0, LENGTH, 0.0)
    curves = [
        geo.addCircleArc(arc_start, center, arc_mid),
        geo.addLine(arc_mid, top_right),
        geo.addLine(top_right, bottom_right),
        geo.addLine(bottom_right, arc_start),
        geo.addCircleArc(arc_mid, center, arc_end),
        geo.addLine(arc_end, top_left),
        geo.addLine(top_left, top_right),
    ]
    lower = geo.addPlaneSurface([geo.addCurveLoop(curves[:4])])
    upper = geo.addPlaneSurface([geo.addCurveLoop([curves[4], curves[5], curves[6], -curves[1]])])
    geo.synchronize()

    for curve in curves:
        gmsh.model.mesh.setTransfiniteCurve(curve, n + 1)
    for surface in [lower, upper]:
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.setRecombine(2, surface)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    element_type = gmsh.model.mesh.getElementType("Quadrangle", order)
    _, _, _, nnodes, local_coords, _ = gmsh.model.mesh.getElementProperties(element_type)
    tags, flat_coords, _ = gmsh.model.mesh.getNodes()
    _, flat_elements = gmsh.model.mesh.getElementsByType(element_type)
    gmsh.model.remove()

    # gmsh node order -> tensor-product order (xi fastest, then eta)
    local_coords = np.round(np.reshape(local_coords, (nnodes, -1))[:, :2], 8)
    tensor_order = np.lexsort((local_coords[:, 0], local_coords[:, 1]))

    order_of_tags = np.argsort(tags)
    coords = flat_coords.reshape(-1, 3)[order_of_tags, :2]
    renumber = np.zeros(tags.max() + 1, dtype=int)
    renumber[tags[order_of_tags]] = np.arange(len(tags))
    elements = renumber[flat_elements.reshape(-1, nnodes)][:, tensor_order]

    # flip clockwise elements, so that every Jacobian is positive
    grid = elements.reshape(-1, order + 1, order + 1)
    origin, right, top = coords[grid[:, 0, 0]], coords[grid[:, 0, -1]], coords[grid[:, -1, 0]]
    edge_a, edge_b = right - origin, top - origin
    cross = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    grid[cross < 0] = grid[cross < 0, :, ::-1]
    elements = grid.reshape(len(elements), -1)

# --------------------------------------- export --------------------------------------
    np.savez(
        DATA_DIR / f"perforated_plate_p{order}_{level}.npz",
        coords=coords,
        elements=elements,
        order=order,
        elements_per_side=n,
    )
    print(f"p {order}  n {n}  {len(coords)} nodes, {len(elements)} quadrilaterals")

gmsh.finalize()
