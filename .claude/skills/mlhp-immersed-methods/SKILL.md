---
name: mlhp-immersed-methods
description: "Concepts, C++/Python API, and debugging advice for immersed / finite-cell (FCM) methods in mlhp: implicit domains, cut-cell quadrature, the alpha parameter, weak boundary conditions, solver conditioning, and postprocessing. Use when setting up, tuning, or debugging an embedded-domain problem in mlhp (C++ or Python)."
---

# mlhp - immersed / finite cell methods

How to set up and debug embedded-domain problems in mlhp. The C++ and Python
APIs differ; both are given per topic with divergences flagged. The
`examples/` scripts (end of file) are the authoritative reference.

## 1. The method and the cut-cell problem

An immersed (fictitious-domain) method embeds the physical domain - given by
an **implicit function** (a predicate: inside -> true) - in a simple
background mesh instead of meshing the geometry. Each background cell is then
*inside*, *outside*, or *cut* by the domain boundary. Cut cells are where all
the difficulty lives. With no stabilization (`alpha = 0`) a cut cell is
effectively *truncated* to its physical part, so its effective element size `h`
can be arbitrarily small - and most of the trouble follows from that one fact:
- **Conditioning.** The stiffness condition number grows like `~1/h^2`, so a
  vanishing effective `h` drives it unbounded (the matrix is outright singular
  for an empty cell).
- **Explicit time step.** The critical step scales with element size, so it too
  shrinks arbitrarily (quantified under explicit dynamics below).
- **High order suffers more than linear.** On a thin physical sliver the
  high-order modes all restrict to nearly linear shapes - nearly linearly
  dependent - so the local basis is close to rank-deficient. The linear modes
  do not degenerate this way (one is ~0, the other ~1 across the sliver), so
  high-order cut cells condition markedly worse.

Many stabilization strategies exist - FCM, CutFEM, eigenvalue stabilization,
element removal, and others. mlhp centers on **FCM** because that is where its
research focus has been, not because of any structural limitation. FCM can
still filter fully-fictitious cells like any method (a choice you make, not
automatic); what is distinctive is that on the remaining (cut) cells it keeps
the weak form over the fictitious part too, scaling that contribution by a
small factor **alpha** rather than dropping it. Often alpha can be folded into
the material parameters, giving the engineering reading of embedding the
structure in a very soft material (Young's modulus scaled by alpha).

Two ways to apply alpha, both valid - but **never both at once**, or the
effective factor becomes `alpha^2`. *Via the quadrature weights* (the default):
scale the fictitious integration weights by `epsilon`, leaving the material
untouched. *Via the material* (`mask`): scale the material parameters outside
the domain and pass `epsilon = 1` to the quadrature. Prefer the quadrature
route - it sidesteps the question of *what* to scale: the `du/dt` term of the
wave equation carries no material coefficient yet still needs scaling, and
Navier-Stokes has no obvious parameter to touch; scaling the weak form avoids
touching the physics at all. (Moment fitting pulls alpha into the weights, so
there you must keep it out of the material.) The difference surfaces in
postprocessing: with the quadrature route, stresses evaluated in the
*fictitious* region come out spuriously high (full material on a
barely-constrained solution) - invisible if you clip output to the physical
domain, visible otherwise; with material scaling they come out small
automatically.

**alpha is the central FCM knob** (`alpha = 0` is no FCM at all - a bare,
unstabilized embedded domain). It is a trade-off, not a value to "get right":
- Lower alpha -> lower modeling error, worse conditioning.
- Higher alpha -> better conditioning, higher modeling error.

The modeling (consistency) error exists because the extended weak form is
inconsistent: the exact solution does not satisfy it. For Poisson/elasticity
the energy error scales like **sqrt(alpha)** on a fixed grid
(https://doi.org/10.1007/s10915-015-9997-3). If you filter outside elements
and keep refining (uniformly), you add an `h^(1/2)` term from the effectively
voxelized geometry, so the error does not level off at `sqrt(alpha)` but keeps
converging at a shallow slope (~1/2). That shallow slope is asymptotic - it
only emerges after enough refinement, and the onset depends on alpha and `p`;
for `p = 1` with element filtering it can take a long time to actually see the
1/2 slope, even at `alpha = 1e-4`. Practically: **alpha = 1e-4 is broadly in
the 1% energy-error range**, often less with filtering; sensitive problems need
lower.

For explicit dynamics (not central in mlhp, though there are examples), the
truncation mechanism above sets the critical time step. FCM bounds it: in the
worst case the cut-cell CFL limit scales with
`cbrt(alpha)` relative to uncut cells
(https://doi.org/10.1007/s00466-025-02704-3). Generalized eigenvalue
stabilization (GEVS) is one option to improve it further
(https://doi.org/10.1016/j.cma.2026.118727), among others.

### C++ vs Python naming (the parts that bite)

| Concept | C++ | Python |
|---|---|---|
| alpha factor | `alpha` (constructor arg) | `epsilon=` keyword |
| space-tree quadrature | `SpaceTreeQuadrature( f, alpha, depth, ... )` | `spaceTreeQuadrature( f, depth, epsilon=, ... )` |
| moment fitting | `MomentFittingQuadrature( f, alpha, depth, ... )` | `momentFittingQuadrature( f, depth, epsilon=, ... )` |
| boolean ops | `implicit::add / intersect / subtract` | `implicitUnion / implicitIntersection / implicitSubtraction` |
| invert domain | `implicit::invert<D>( f )` | `invert( f )` |
| simplex-mesh filter | `filterSimplexMesh( mesh, f )` (free fn) | `mesh.filter( f )` (method) |

Note the quadrature arg order differs (alpha/depth are swapped) *and* alpha is
called `epsilon` in Python - the two easiest mistakes to make.

## 2. Defining the domain

A domain is an `ImplicitFunction<D>` - a predicate
`std::array<double,D> -> bool` (inside -> true). Ways to build one, roughly by
how often they matter:

- **Arbitrary custom domain.** In C++ you can write a lambda returning `bool`.
  In Python, `implicitFunction( ndim, ... )` uses the same expression-string
  parser as `scalarField` and is very powerful; it also accepts a compiled
  function by address, so even C-compiled domains work (see
  `poisson_compiled.py` / `j2_plasticity_compiled.py`).
- **STL surfaces.** `readStl` + `rayIntersectionDomain` turn a triangulated
  surface into a domain.
- **Voxel data.** `scalarFieldFromVoxelData` (Python) / `spatial::voxelFunction`
  (C++) build a field from a voxel grid; threshold it (`implicitThreshold`) to
  get a domain - the usual route for image-based geometry.
- **CSG primitives + booleans.** `implicit::sphere`/`ellipsoid`/`cube`/
  `halfspace` (Python `implicitSphere`/... ) combined with `add`/`intersect`/
  `subtract`/`invert` (Python `implicitUnion`/`implicitIntersection`/
  `implicitSubtraction`/`invert`), plus `extrude`/`transform`. For simple
  analytic domains and for combining any of the above.

Cut classification samples `nseedpoints` per direction (default 5) to decide
inside/outside/cut.

### Filtering the background mesh

Dropping fully-outside cells before assembly (fewer DOFs) is done with
`makeFilteredGrid` (Python) / `FilteredGrid` (C++). Filtering is **seed-point
based** by default, like the cut classification above.

That seed-point basis has a subtlety: space-tree quadrature also classifies
intersection by seed points and *then* distributes Gauss points, so a cell the
seed grid calls "cut" can end up with no Gauss point in the physical part at
all. With a decently high alpha this is usually harmless - but it bites when
that sliver carries a boundary condition, because you are then imposing a BC on
an element with no physical integration point for it to act on. A polished
quadrature-based filter (comparing the known cell measure against the
integrated measure) does not exist yet; it can be done by hand in C++ or Python
(e.g. via the quadrature `evaluate` function), but that is inelegant and slow.

## 3. Cut-cell quadrature

Two main schemes, both taking the domain, alpha (`epsilon`), a refinement
`depth`, and `nseedpoints`:

- **Space tree** (`SpaceTreeQuadrature` / `spaceTreeQuadrature`): recursively
  bisects cut cells toward the boundary, tensor-product Gauss in each
  sub-cell. Robust default - correct whenever the integrand jumps at the
  boundary but is otherwise smooth.
- **Moment fitting** (`MomentFittingQuadrature` / `momentFittingQuadrature`):
  fixes the weights to integrate a polynomial exactly - `2p+1` points for
  degree `2p` (mlhp defaults to `2p+2` for safety, e.g. a linear material).
  Far fewer assembly points.

**Which to use** (challenge -> choice):
- Integrand is polynomial *and* you need more than one space-tree level (you
  almost always do): use **moment fitting**. With a single space-tree level
  the point counts match, so prefer space tree there (more robust, no weight
  computation).
- Integrand has reduced regularity (plasticity) or a discontinuity (e.g. an
  L2 projection whose RHS is scaled to zero in the fictitious region -> step
  function): use **space tree**. Moment fitting assumes a polynomial; here it
  can silently produce wrong integrals, or work until you tweak the setup and
  then break. When in doubt, space tree.
- Space tree too expensive: it can emit a huge number of points, each costing
  a `B^T C B` (worse at high `p`, which also needs more depth to resolve
  geometry). When assembly time is the bottleneck, moment fitting may be the
  only feasible option.

**Niche alternatives (good to know, rarely needed):**
- **Cell-mesh quadrature** (`CellmeshQuadrature` / `cellMeshQuadrature`, fed by
  `domainCellMesh`): does a volumetric marching-cubes recovery of the domain -
  originally a postprocessing tool - and integrates over that mesh.
  `domainCellMesh` can mesh *both* sides, so the fictitious part can be
  integrated with alpha. Not the standard route and not used by default.
- **Grid quadrature** (`GridQuadrature` / `gridQuadrature`): uniform sub-cell
  subdivision. The space-tree seed-point test can, very occasionally, miss a
  thin feature; the more uniform sampling of grid quadrature can then slightly
  improve boundary stress artifacts. Unusual, but worth knowing.

**Depth heuristic**: start around **p + 1** levels (same for space tree and
the tree inside moment fitting). It is geometry-dependent - some cases need
far more (wave propagation convergence has needed ~16 levels in 2D). Convergence 
plots show an early, zigzaggy level-off for few levels (often good enough in
practice); true energy-norm convergence can require a *lot* of refinement.

> Trap: moment fitting on a non-polynomial or discontinuous integrand. The
> symptom is wrong results with no error thrown. Diagnosis: re-run that case
> with space tree; if the answer moves, the integrand was not polynomial. Fix:
> use space tree.

## 4. Boundary conditions

Not every boundary is immersed. Sides that conform to the background-mesh
facets get their boundary conditions the normal (strong/conforming) way -
prefer that wherever it applies; only the genuinely immersed boundary needs
weak imposition. A facet that is *part* physical and *part* fictitious can
still use the conforming route: for Dirichlet this is fine (possibly a small
accuracy hit); for Neumann you must **mask the applied load** so nothing is
imposed over the fictitious part.

For the immersed boundary, impose weakly: recover it as a simplex surface mesh
(`recoverDomainBoundary`), then integrate a boundary integrand over it
(`integrateOnSurface`).

**Penalty is the default first try** (`l2BoundaryIntegrand` in Python /
`makeL2BoundaryIntegrand` in C++): simple, and usually works well. Two
gotchas:

- **Conditioning**: a large penalty worsens conditioning. Diagonal (Jacobi)
  preconditioning alleviates this a lot.
- **Scaling**: the penalty is a distributed spring *stiffness* with a real
  physical interpretation - it must be scaled to the magnitude of the K
  diagonal, not assumed to be O(1).
  - Diagnosis: print the K diagonal min/max/mean and size the penalty to it.
- **Locking** (the subtle one): a high penalty on a cut element can lock the
  whole element to zero. E.g. a bilinear element with a curved immersed
  Dirichlet boundary - no nonzero combination of the bilinear shapes satisfies
  the constraint, so the element is pinned.
  - Symptom: solution collapses to ~0 over cut elements.
  - Fix: lower the penalty until it permits local deviation, or go to higher
    order (much harder to fully lock - at the cost of worse conditioning).

Nitsche is the principled alternative (it folds the physics into the boundary
term and balances domain vs boundary contributions, which penalty does not). 
One caveat to revisit once it is supported: the Nitsche parameter can grow 
unbounded for certain cut configurations.)

## 5. Solver conditioning

**First move**: run `p = 1` with a high `alpha`. Even if the solution is
poor, it tells you whether the setup works at all.

Isolate formulation bugs from conditioning:
1. Dense direct solve on a tiny mesh to validate the formulation.
2. Set `alpha = 1`, `penalty = 0` to recover the clean body-fitted baseline,
   then add complexity back **one knob at a time**.
3. Print the K diagonal to scale the penalty (section 4).

**Solvers in mlhp**: only `cg`, `bicgstab`, and the dense direct solver - by
design, to avoid dependencies (application code is separate and can pull in
more). Use diagonal/Jacobi preconditioning. Additive Schwarz exists but, in
the element-wise form implemented, often performs worse than Jacobi (roughly
2x cost per iteration); the version that actually helps inverts larger groups
of elements (https://doi.org/10.1016/j.finel.2019.01.009), which mlhp does not
have yet. Incomplete-LU has not worked here (anecdotal).

**Escape hatches when iterative solvers stall** (downstream / application
code, not mlhp core):
- **Pardiso (Intel MKL)**: `src/python/mklwrapper.py` in the repo is a thin
  Python wrapper (`pip install mkl`; not shipped in the wheel). Handles FCM to
  hundreds of thousands of DOFs comfortably, millions given time (avoid inside
  a many-step time loop). Can also be linked from C++ (uglier; fine downstream,
  not in mlhp itself).
- **scipy `spsolve`** (Python): convenient but struggles past ~100k DOFs.
- **GPU solve** (Python; commented code in the elastic gyroid example): 20x+
  speedups for ~100k-30M DOFs - brute force past bad conditioning. Below ~100k
  the GPU is underused and transfer dominates. Double precision runs
  surprisingly fast; single precision may be insufficient.

## 6. Postprocessing

Restrict output to the physical material - the background mesh covers the
fictitious region too:
- `cellmesh::domain` / `domainCellMesh` (marching-cubes interior),
  `recoverDomainBoundary` for the surface, `simplexMesh.filter` to drop
  outside simplices.

**Stresses (and other gradients) can be arbitrary in small cut cells** - this
is inherent to the method, not necessarily a bug. A tiny corner may have no
integration point at all, so the local solution and its gradient are
meaningless there. Options:
- Nudge/rotate the domain slightly - a cosmetic quick fix for a clean image,
  not a real solution.
- Project the displacement gradient onto the basis and compute stresses from
  that (mlhp has this). It smears stress jumps and tames corner artifacts, but
  does not remove them entirely and can look smeared. (The raw jumps are
  honest - sometimes preferable.)

## Reference examples

Authoritative end-to-end setups in `examples/`:
- `elastic_fcm_csg.py` - 3D elasticity on a CSG domain; implicit construction,
  boundary recovery, penalty Dirichlet, space-tree quadrature, boundary output.
- `planestress_fcm_plate_with_hole.py` - 2D plane stress; moment fitting,
  `domainCellMesh`, adaptive refinement, stress output.
- `planestress_fcm_lshaped_adaptive.py` - rotated L-shape; `implicitTransformation`,
  moment fitting, stress-jump-driven refinement.
- `elastic_fcm_stl.py` - 3D elasticity on STL geometry via `rayIntersectionDomain`.
- `wing_elastic_fcm.cpp`, `linear_elastic_fcm_stl.cpp`,
  `j2_pressurized_sphere_fcm.cpp` - C++ equivalents (incl. nonlinear J2).