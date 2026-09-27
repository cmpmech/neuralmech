# Voxel Multigrid on the GPU

Matrix-free finite elements on voxel data for **Chapter 18 (Simulation Acceleration
via GPUs)**: elements preintegrated per sub-voxel with `mlhp`, solved with
multigrid-preconditioned conjugate gradients in CuPy so that the iteration count stays
flat as the problem grows. `helper.py` holds the solver (`VoxelMultigrid`), meant to
move to `solvers/` once settled.

## Drivers

- `elastic_forward.py`
  linear elastic gyroid cantilever at growing voxel resolution (`--dim 2` or `3`),
  reporting dofs, CG iterations and times per resolution
- `topopt_elasticity2D_mgcg.py`
  the compliance topology optimization of `8_physics_drivers/topopt_elasticity2D.py`
  (half MBB beam, optimality criterion) with the GPU multigrid solver in place of
  pardiso

## Non-obvious technicalities (authored by Claude)

**Preintegration and the matrix-free operator.** A fine element spans $s^D$ voxels.
`mlhp.integratePartitionMatrices` integrates one unit-coefficient matrix $K_s$ per
sub-voxel once, so the element matrix of any design is $K_e = \sum_s c_{e,s} K_s$ and
the operator only reads the voxel coefficients $c$. The kernels work row-wise: thread
$i$ sums row $i$ over the at most $2^D$ elements holding dof $i$, found through a
precomputed dof-to-element map. No atomics are needed, every sum runs in a fixed order
(runs are bitwise reproducible), and a whole Chebyshev step (matvec, residual, search
direction and iterate update) fuses into one kernel launch.

**Hierarchy.** For $p > 1$ the first coarsening drops to $p = 1$ on the same mesh, then
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

**Where it still struggles.** Slender curved members separated by near-empty void
(the 2D gyroid slice with void stiffness $10^{-6}$) need about 150 iterations,
independent of resolution, against 14 for the full solid. Bilinear coarse functions
cannot let neighbouring members, split by a gap narrower than a coarse cell, move
independently. Operator-dependent coarsening (smoothed aggregation) would be the next
step. Optimized SIMP designs and the connected 3D gyroid converge in 25 to 50 cold
iterations, and in about 13 inside the optimization with warm starts.
