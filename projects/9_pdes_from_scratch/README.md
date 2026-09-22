# PDE solvers from scratch

The three discretizations of **Chapter 9 (Numerical Methods)** written out directly
instead of called from a library. All of them solve the same Poisson problem,
$-k\Delta u = p$ with $\nabla u\cdot\mathbf{n}=f$ on the right edge and $u=0$ on the
remaining outer edges and on the hole rim, on a square plate with a circular hole and a
Gaussian source beside it.

## Drivers

`create_mesh.py -> data/plate_with_hole.npz`
    quadrilateral mesh of the plate with a hole, cut and recombined with gmsh;
    `LENGTH`, `HOLE_RADIUS` and `ELEMENT_SIZE` are the knobs

`fem.py`
    Galerkin finite elements with bilinear shape functions, an isoparametric map to the
    biunit square and 2x2 Gauss quadrature, showing the nodal field on the mesh
    _needs plate_with_hole.npz_

`fvm.py`
    cell-centred finite volumes with a two-point flux across each face, showing one
    constant value per cell
    _needs plate_with_hole.npz_

`fdm.py`
    the five-point stencil on a uniform Cartesian grid, which ignores the mesh and
    staircases the hole instead
    _needs plate_with_hole.npz_

The three solvers share `CONDUCTIVITY`, `FLUX` and the source settings `AMPLITUDE`,
`SOURCE_CENTER` and `WIDTH`, so changing a number in one file and the same number in
the others compares the methods on the same problem. `RESOLUTION` refines the finite
difference grid, `ELEMENT_SIZE` the mesh the other two read.

## Non-obvious technicalities (authored by Claude)

The element integrals are written as an explicit quadrature loop rather than as a
precomputed matrix product, so that a different weak form is a local edit. In `fem.py`
the two lines

```python
Ke += CONDUCTIVITY * dN_dx.T @ dN_dx * weight
fe += N * source(N @ xe) * weight
```

are the whole integrand of $K^e_{ij}=\int_{\Omega^e} k\nabla N_i\cdot\nabla N_j$ and
$f^e_i=\int_{\Omega^e} N_i p$; everything around them is geometry and bookkeeping that
any second-order scalar problem shares. The Jacobian convention is the one place where
the index order matters: `jac = dN_dxi @ xe` stores $\partial x_a/\partial\xi_b$ at
`jac[b, a]`, so the chain rule reads `dN_dxi = jac @ dN_dx` and the physical gradients
follow from `np.linalg.solve(jac, dN_dxi)` rather than from an explicit inverse.

The geometry parameters ride along inside `plate_with_hole.npz` instead of being
imported from `create_mesh.py`. Drivers here have no `__main__` guard, so importing the
mesh script would re-run the mesher; saving `length`, `hole_center` and `hole_radius`
next to the connectivity lets `fdm.py` rebuild the level set in one line with no
coupling between the files.

The Neumann edge is the only boundary condition that looks different in each method,
which is why it is there. The finite element driver evaluates the boundary integral
$\int_{\Gamma_N} N_i k f \,d\Gamma$ with a one-dimensional Gauss loop over the edges
that appear only once in the connectivity. The finite volume driver adds the prescribed
flux straight to the right-hand side, with no approximation at all, because the flux
across a face is exactly what the scheme balances. The finite difference driver
replaces the normal derivative by a one-sided difference, as the chapter states, which
is the cheapest of the three and the only first-order one -- the second-order
alternative is a ghost node eliminated through the central difference, two lines more.
The Dirichlet parts, by contrast, are identical bookkeeping in all three.

`fdm.py` keeps every grid node as an unknown and gives the nodes inside the hole an
identity row, which is the ordinary Dirichlet treatment applied to a blob-shaped set of
nodes. The consequence is visible in the figure: the rim of the hole is a staircase of
grid cells rather than a circle. The constrained cells are drawn without faces and
without grid lines (`pcolormesh` takes one edge color per cell, so the lines inside the
hole are simply transparent), which makes the staircase the visible difference between
the three pictures, and it costs accuracy. `RESOLUTION` is set so that the grid carries
roughly as many degrees of freedom as the mesh the other two read (1156 against 1047
and 1098, which each driver prints), and at that budget the staircase shows: the value
at the source is 0.201 against the 0.190 of the finite element solution. Refining to
241 nodes per side brings it to 0.190 as well.

All four figures draw the mesh with the same `MESH_COLOR` and `MESH_WIDTH`, but
`fdm.py` has to pass `antialiased=True` explicitly: `pcolormesh` defaults to `False`,
unlike `PolyCollection`, and without antialiasing a sub-pixel line width cannot be
drawn faintly -- it snaps to a solid one-pixel line at full opacity, which made the
finite difference grid look several times heavier than the meshes at the same setting.
`create_mesh.py` is the one figure with nothing behind the lines, so it darkens the
colour to stay legible on white while keeping the same width.

Face conductivities are absent from `fvm.py` because $k$ is constant here. With a
varying coefficient the two-point flux needs $k$ on the face rather than in the cell,
and the harmonic mean is the right average: it keeps the flux continuous across a jump
in $k$, whereas the arithmetic mean does not.
