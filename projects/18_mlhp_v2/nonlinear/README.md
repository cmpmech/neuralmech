# Nonlinear Voxel Multigrid (exploration)

Extensions of `../helper.py` towards nonlinear problems, kept separate until settled:
`helper.py` here is a superset of the linear solver (one operator family reproduces
it). Both drivers use the same multigrid-preconditioned CG on the GPU.

## Drivers

- `phasefield_notched2D.py`
  AT2 phase-field fracture of the notched square (`--load tension` or `shear`,
  Miehe et al. 2010) with the volumetric-deviatoric split and a staggered scheme
- `hyperelastic_gyroid.py`
  compression of a voxelized gyroid bar (`--dim 2` or `3`) with a compressible
  neo-Hookean material written as a GPU material subroutine, solved by Newton

## Non-obvious technicalities (authored by Claude)

**Operator families.** The linear solver scales one preintegrated sub-voxel matrix by a
voxel coefficient. Many nonlinear problems keep that structure with a few families,
$K_e = \sum_t \sum_s c_{t,e,s} K_{t,s}$: the kernels simply treat $t$ as more
sub-voxels, and the Galerkin coarse operators stay a single gemm. `QuadratureTables`
builds any family from the shape functions at the Gauss points of one element (all
voxel elements are translates), e.g. $\sum_q w_q B_q^\top C B_q$ for elasticity or
rank-one sub-voxel means; they reproduce the `mlhp` element matrices to machine
precision.

**Phase-field fracture fits the families exactly.** With the history field $H$ the
staggered subproblems are linear. The phase field solves
$G_c (M/\ell + \ell L) d + \sum_s 2 H_s \bar M_s d = \sum_s 2 H_s |s| \bar m_s$, two
families with coefficients $G_c$ and $2H_s$. For the displacement, the volumetric
split is taken on the sub-voxel mean dilatation $\bar\varepsilon_v = \bar b_s \cdot u_e$:
$\psi = g\,\psi_{dev} + g\,\psi_{v}' + (g\,[\bar\varepsilon_v > 0] + [\bar\varepsilon_v < 0])
\tfrac{\kappa}{2}|s|\bar\varepsilon_v^2$, with $\psi_v'$ the dilatation fluctuation. The
energy is piecewise quadratic and $C^1$, so $K(c(u))\,u$ is its exact gradient, the
switch pattern is resolved by a semismooth Newton (usually one or two solves) and
the families are $K$ and the rank-one $\kappa |s| \bar b \bar b^\top$ with coefficients
$g$ and $(1 - g)[\bar\varepsilon_v < 0]$. Lumping the phase-field mass (with $g$ averaged
over the nodes) makes the phase-field matrix an M-matrix, so $0 \le d \le 1$ without
clipping (the consistent mass overshoots to about 1.15), but at $\ell/h \approx 2$ it
biases the crack towards the grid axes (the shear crack then runs horizontally);
$\ell/h \approx 4$ recovers the curved shear crack.

**General materials.** Where the tangent is not coefficient-linear (finite strain,
plasticity, spectral splits), `PointMaterial` compiles a CUDA material subroutine
$(\nabla u, \text{parameters}) \mapsto (P, \partial P/\partial F)$, the GPU analogue of the
`mlhp` cffi/numba materials. Forces and element tangents follow from the same tables
and the multigrid runs on stored element matrices. The tangents are formed in float32
(they only steer Newton, the residual stays float64), which is four times faster on a
consumer GPU without changing the Newton iterates. Storage is $n^2$ per element (576
doubles for trilinear hexahedra), which bounds the problem size in 3D.

**Lazy hierarchy.** Between staggered or Newton iterations only the float64 CG operator
is updated (`hierarchy=False`); the V-cycle of the last full setup stays. The CG is
already flexible, and the hierarchy is rebuilt from the current operator only once a
solve exceeds `refresh` iterations. Rebuilding costs as much as several CG iterations.

**Staggered beats monolithic here.** The monolithic Jacobian is matrix-free from the
same families, the couplings being the sub-voxel forces $q_s = K_s u_e$ scaled by
$g'(d)$. With the history active it is the Hessian of the non-convex AT2 energy and
indefinite, so GMRES needs 40 to 70 iterations per Newton step even with exact block
solves. BFGS around the block-diagonal operators avoids indefinite solves (6 to 8
iterations per step), but each iteration costs more than a staggered one, and both
monolithic schemes diverge at the brittle snap, which the energy-decreasing staggered
iteration passes. All three agree on the tension peak (0.78 kN at 5.94e-3 mm, 256
voxels). Tried on the staggered scheme without a net gain: adaptive load steps (13x
fewer steps, but the staggered iterations track the damage evolution and stay near
4000), extrapolating $d$ between steps (18% fewer iterations, costlier solves), a
looser staggered tolerance, and a p = 1 stencil kernel (the row kernels are float64
bound, not gather bound, so only 10 to 20% faster). A CG tolerance of $10^{-4}$ is
fast but wrong: warm-started solves then stall and the peak rises by 4%.

**Where it struggles.** Broken voxels under compression keep only the stiff rank-one
dilatation constraint without shear stiffness, a locally incompressible material on
which geometric multigrid degrades (hundreds of iterations in a random stress test,
rare in the benchmarks). The history-field staggered scheme needs hundreds of
iterations per step during stable crack growth; Anderson acceleration of the phase
field alone made it worse. The hyperelastic gyroid walls buckle beyond about 12%
compression, which plain Newton under displacement control cannot follow.
