# Voxel Multigrid on the GPU

Matrix-free finite elements on voxel data for **Chapter 18 (Simulation Acceleration
via GPUs)**: elements preintegrated per sub-voxel with `mlhp`, solved with
multigrid-preconditioned conjugate gradients in CuPy so that the iteration count stays
flat as the problem grows. `helper.py` holds the general solver (`VoxelMultigrid`, any
`mlhp` basis), `structured.py` a lean 2D variant for large voxel grids
(`StructuredMultigrid`); both are meant to move to `solvers/` once settled.

## Drivers

- `elastic_forward.py`
  linear elastic gyroid cantilever at growing voxel resolution (`--dim 2` or `3`),
  reporting dofs, CG iterations and times per resolution
- `topopt_elasticity2D_mgcg.py`
  the compliance topology optimization of `8_physics_drivers/topopt_elasticity2D.py`
  (half MBB beam, optimality criterion) with the GPU multigrid solver in place of
  pardiso
- `topopt_elasticity2D_structured.py`
  the same optimization with the structured-grid solver, for tens of millions of
  voxels on a single GPU

## Non-obvious technicalities (authored by Claude)

**Preintegration and the matrix-free operator.** A fine element spans $s^D$ voxels.
`mlhp.integratePartitionMatrices` integrates one unit-coefficient matrix $K_s$ per
sub-voxel once, so the element matrix of any design is $K_e = \sum_s c_{e,s} K_s$ and
the operator only reads the voxel coefficients $c$. The kernels work row-wise: thread
$i$ sums row $i$ over the at most $2^D$ elements holding dof $i$, found through a
precomputed dof-to-element map. No atomics are needed, every sum runs in a fixed order
(runs are bitwise reproducible), and a whole Chebyshev step (matvec, residual, search
direction and iterate update) fuses into one kernel launch.

**Hierarchy.** For $p > 1$ the degree drops by one per level on the same mesh, then
every further level halves the mesh until at most `coarse_dofs` remain, which are
solved with a dense inverse. The element-local transfer maps $T_k$ (coarse element dofs
to the dofs of child $k$) come from a least-squares fit of the coarse basis in the fine
one on sample points, which is exact for nested spaces and independent of the shape
function family (for the hierarchical `mlhp` bases, $p \to 1$ reduces to injection).

**Galerkin coarse operators as one gemm.** The coarse element matrices are the exact
Galerkin products $K_E = \sum_k T_k^\top K_k T_k$. With
$\mathrm{vec}(T^\top K T) = (T \otimes T)^\top \mathrm{vec}(K)$, all children collapse
into a single matrix product of the stacked child matrices with
$G = [T_1 \otimes T_1; \dots]$. Galerkin operators (not rediscretized ones) keep the
coarse levels consistent with $10^{-9}$ SIMP contrast. Level 1 is formed directly from
the voxel coefficients with the preintegrated $\mathrm{vec}(T_k^\top K_s T_k)$, so the
fine element matrices are never stored; only the few coarse elements whose children
touch Dirichlet dofs are recomputed with masked child matrices.

**Dirichlet conditions.** Fixed dofs act as identity rows, $A = M K M + (I - M)$ with
the 0/1 mask $M$. Coarse dofs whose Galerkin diagonal vanishes (they only touch fixed
fine dofs) are masked in turn. Every vector in CG and the V-cycle stays zero on fixed
dofs, so the kernels only mask the output row.

**Element-block Schwarz on the $p > 1$ levels.** Once part of a higher-order element
is void, its polynomials combine into near-zero-energy modes that span the element
(the finite cell ill-conditioning). Jacobi cannot damp them and a lower degree cannot
represent them: $p = 3$ with $4^2$ sub-voxels needed about 290 CG iterations. The
$p > 1$ levels therefore use Chebyshev with an additive Schwarz preconditioner whose
blocks are the assembled matrix restricted to each element's dofs, inverted per design
after diagonal scaling (float32 suffices, float64 inverses are slow on consumer GPUs).
This restores 10 to 15 iterations.

**Smoother and precision.** Chebyshev-Jacobi smoothing on
$[\lambda_{\max}/30, 1.1\,\lambda_{\max}]$ with $\lambda_{\max}(D^{-1}A)$ from a
warm-started power iteration. The degree grows on the coarse levels (default 2, 4, 8,
16), which is where thin members are hardest to represent and where extra steps are
cheap. The V-cycle runs in float32 inside a float64 CG; the float32 preconditioner is
not exactly linear, so CG uses the flexible (Polak-Ribiere) $\beta$. Running CG itself
in float32 is not viable: its attainable accuracy $\epsilon_{32}\,\kappa(A)$ gave a 0.3%
error at void stiffness $10^{-6}$ and a wrong SIMP design at $10^{-9}$, and float32
iterative refinement either stalls ($\kappa(A)\,\epsilon_{32} > 1$) or saves nothing.

**One CUDA graph per CG iteration.** At these sizes kernel launches, not GPU work,
dominate a V-cycle. All vectors, scalars, CSR transfers and the coarse inverse are
preallocated and updated in place, dot products and transfers are custom kernels, so
one full CG iteration (V-cycle included) is captured once and replayed as a single
graph launch.

**Higher order with sub-voxels does not pay off here.** Groen et al. (2017) decouple the
design voxels from higher-order analysis elements, with $p \ge \mathrm{round}(0.75\,s)$
against artificially stiff patterns, and gain about 3x in 2D with a direct solver. For
the half MBB beam at 640 x 160 voxels, $p = 3$, $s = 4$ (trunk space, a third of the
dofs) took 23 s against 16 s for $p = 1$, $s = 1$, and 126 s against 92 s at
1280 x 320, at the same reference compliance and without visible artefacts. The
$p = 1$ voxel operator stores no element matrices (one coefficient per voxel and an
8 x 8 matrix in cache), whereas higher order streams stored element matrices and
Schwarz blocks and rebuilds them per design; the multigrid already makes the solve
cost independent of resolution, which is what a direct solver lacks. Full tensor
spaces at $p = 4$ still need hundreds of iterations.

**Structured grids for tens of millions of voxels.** With `VoxelMultigrid` only about
100 of its roughly 500 bytes per voxel held vectors that CG and the V-cycle need; the
rest were generic mesh data (element-to-dof and dof-to-element tables, sparse transfer
matrices, stored coarse element matrices, float64 masks). `StructuredMultigrid` drops
all of it for bilinear voxel elements: dofs are ordered (field, i, j), one thread per
node finds its four elements and nine neighbouring nodes by index arithmetic,
prolongation is bilinear interpolation and restriction its transpose, both as
stencils. The three finest levels are applied from the voxel coefficients alone,
level $d$ as $\sum_k c_k T_k^\top K T_k$ over the $4^d$ voxels of each element with
the preintegrated $T_k^\top K T_k$; this costs as many flops per voxel as the fine
level and reads nothing but $c$. The next level stores its Galerkin matrices formed
straight from the voxels, coarser ones from their children. Coarse Dirichlet masks are
injected from the coincident fine node, which reproduces the masked Galerkin operator
exactly when fixed dofs sit on coarse nodes or cover whole faces (identical iteration
counts for the MBB beam). CG keeps $x$ and $r$ in float64 but $p$, $Ap$ and $z$ in
float32; $Ap$ is still accumulated in float64, since float32 arithmetic there makes CG
diverge at void stiffness $10^{-9}$ even with periodic float64 residual replacement,
and the compliance then agrees with the float64 solver to about $4 \cdot 10^{-7}$.
With a fused optimality-criterion update this comes to about 205 bytes per voxel and
2.5 times less time per design iteration than `VoxelMultigrid` (2560 x 640 voxels:
0.29 s against 0.72 s), where the float64 fine operator alone is a quarter of every CG
iteration on consumer GPUs with little float64 throughput.

**Bilinear elements over sub-voxels.** `sub_voxels` lets one bilinear element span
$s \times s$ voxels while density, sensitivities and filter stay per voxel (the
energy of voxel $k$ is $u_e^\top T_k^\top K T_k u_e$). Applied from the voxel
coefficients, the operator costs the same flops per voxel for every $s$, so fewer dofs
only shrink the vectors. From $s = 3$ on the fine level therefore uses that the
element matrix of a bilinear element depends on the material only through six moments,
$K_e = \sum_p m_p B_p$ with $m_p = \int_e c\, p$ for $p \in \{1, x, y, x^2, y^2, xy\}$,
since products of bilinear shape function derivatives span exactly these. The $B_p$
come from an exact least-squares fit, six float32 moments per element replace 64
matrix entries, and because every $B_p$ annihilates translations, rounding the moments
only perturbs the material, whereas rounding assembled matrix entries to float32 adds
spurious stiffness against rigid motion. The filter radius has to cover at least one
element: $s = 4$ with $r_{\min} = 2$ voxels gives the classical artificially stiff
node-dot patterns (85% stiffness overestimate on a one-voxel-per-element reference),
while $r_{\min} \ge s$ reproduces the one-voxel designs within 0.1% at a uniform
1 to 2% discretization overestimate. With the filter convolved by FFT (a direct
convolution costs $O(r_{\min}^2)$ per voxel) and a log-space bisection for the
optimality criterion, the 8192 x 2048 half MBB beam ($r_{\min} = 25.6$) takes 49 s
with $s = 8$, 118 s with $s = 4$, 497 s with $s = 2$ and 1101 s with $s = 1$. On
7680 x 1920 voxels ($r_{\min} = 24$) the time per design iteration falls from 0.61 s
($s = 3$) to 0.25 s ($s = 5$) and 0.19 s ($s = 6$), and the thresholded designs of
$s = 3$ to 8 are visually identical and agree within 0.02% on a one-voxel-per-element
reference, while their own compliance drops from 1.2% to 2.2% below it. Grid sizes
should coarsen to a small grid: an odd 60 x 15 element coarse grid (1952 dofs) costs
a float64 dense inversion per design iteration and doubled the time of $s = 4$ and 8.
14336 x 3584 voxels (51 million) fit a 6 GB laptop GPU at about 100 bytes per voxel
with $s = 4$, and $s = 8$ needs 73.

**CG restarts against stagnation.** Occasionally a solve in the optimization ran into
the 500-iteration cap while the replayed system converged in 35. The Chebyshev bounds
were not the cause (the warm-started power iteration was within 2% of the converged
$\lambda_{\max}$); the coarse matrices are: with $10^{-9}$ void their inverse reaches
$5 \cdot 10^{9}$, so float32 rounding of the solid residual turns into huge void
displacements in $z$. Through $p^\top A p$ these shrink the step length, the
preconditioner is no longer quite symmetric (relative error $10^{-3}$), and
flexible CG loses conjugacy and sits on a flat residual for hundreds of iterations,
depending on the rounding history. `solve` therefore restarts from the true residual
when the best residual norm drops less than 2 times over 20 iterations, which costs
one iteration. On the 51 million voxel run this took the worst solve from 500 to
181 iterations and the mean from 26.9 to 22.2; healthy solves never trigger it.

**Where it still struggles.** Slender curved members separated by near-empty void
(the 2D gyroid slice with void stiffness $10^{-6}$) need about 150 iterations,
independent of resolution, against 14 for the full solid. Bilinear coarse functions
cannot let neighbouring members, split by a gap narrower than a coarse cell, move
independently. Operator-dependent coarsening (smoothed aggregation) would be the next
step. Optimized SIMP designs and the connected 3D gyroid converge in 25 to 50 cold
iterations, and in about 13 inside the optimization with warm starts.
