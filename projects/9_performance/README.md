# 9_performance

Performance metrics of numerical methods for Chapter 9, on a quarter plate with a hole
under uniaxial tension.

- `create_meshes.py -> data/perforated_plate_p{order}_{level}.npz` -- conforming
  quadrilateral meshes from gmsh, curved (isoparametric) for order > 1
- `convergence.py` -- energy-norm convergence of h-refinement at p = 1 and p = 3 and of
  p-refinement on 2 elements, with the x displacement at three stages of each study;
  _needs the meshes of create_meshes.py_

## Non-obvious technicalities (authored by Claude)

The meshes consist of two transfinite patches split along the diagonal, so every cell is
a quadrilateral (gmsh's recombination leaves triangles on this geometry). The gmsh node
order is permuted into tensor-product order (xi fastest) using the reference node
coordinates of the element type, and clockwise elements are flipped. gmsh places the
high-order nodes on the circle, so the geometry error falls with the order; straight-sided
cubic elements would cap $p = 3$ at the rate of $p = 1$. gmsh quadrilaterals end at
order 10. mlhp's unstructured basis is low-order only, so the elements are assembled by
hand with equidistant Lagrange polynomials.

The energy error is computed from strain energies alone,
$e_E = \sqrt{|U - \hat{U}|/U}$, which holds for loading by prescribed tractions. The
reference $U$ is the $p = 3$ solution on the mesh of level 6 (148k degrees of freedom);
levels 5 and 6 differ by $3 \cdot 10^{-6}$ in energy. The overkill value of the mlhp
`planestress_fcm_plate_with_hole.py` example is about $4 \cdot 10^{-5}$ too high and would
floor the $p = 3$ curve near $3 \cdot 10^{-5}$. Below a few $10^{-6}$ in $e_E$, the energy
difference reaches round-off: $p$-refinement on 32 elements became non-monotone beyond
$p = 6$.
