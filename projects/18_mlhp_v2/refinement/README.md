# Refined Multigrid for Phase-Field Fracture (exploration)

Local refinement for the phase-field fracture drivers in `../nonlinear/`, kept separate
until settled: bilinear elements on a 2:1 balanced quadtree with hanging nodes, the
same preintegrated operator families and a multigrid-preconditioned CG on the GPU.

## Drivers

- `phasefield_refined2D.py`
  notched tension or shear (`--load`) on a mesh refined around the expected crack
  path, to validate against the uniform `../nonlinear/phasefield_notched2D.py`
- `phasefield_adaptive2D.py`
  the same problems on a mesh that refines ahead of the growing crack

## Non-obvious technicalities (authored by Claude)

**Preintegration survives refinement.** Every leaf is a scaled copy of the unit
square, so the unit-square family $K_t$ holds the whole geometry and a leaf of size $h$
contributes $h^{p_t} K_t$ ($p_t = 0$ for gradient forms in 2D, $p_t = 2$ for masses). A
hanging corner interpolates its constraining nodes, so every element matrix is
$T_e^\top (c_{t,e} h_e^{p_t} K_t) T_e$ with a small constraint matrix $T_e$, and the
global CSR values are a fixed sparse map of the coefficients,
$\mathrm{vec}(A) = M c$, built once per mesh. Any coefficient update, as in every
staggered iteration, is one sparse product. Values stay nodal (unlike the hierarchical
multi-level hp basis of mlhp), so the lumped phase-field mass remains an M-matrix and
fields transfer between meshes by interpolation.

**Multigrid on truncated trees.** Replacing every leaf deeper than level $j$ by its
ancestor gives a coarser balanced tree whose space is nested in the finer one. The
prolongation interpolates the coarse function at the fine nodes, the coarse operators
are Galerkin products $P^\top A P$, and truncation continues down to a few hundred dofs
for a dense solve. Away from the crack the levels coincide with uniform coarsening,
near it they shrink the refined region one level at a time.

**Latency, not flops.** At some ten thousand elements the GPU is idle most of the time,
so the whole flexible CG iteration including the V-cycle is one captured CUDA graph of
fused CSR kernels (Chebyshev degree 2 on every level is the fastest). Element-wise
reductions go through precomputed CSR maps: a cupy `sum` over a short last axis took 7 ms
per call, 200 times the gather itself.

**Adaptive refinement.** After every staggered iteration, elements with a phase field
above a threshold that are not yet at the finest level trigger a refinement of all
leaves within a buffer radius. Refinement only ever adds leaves, so the old space is
nested in the new one and displacement and phase field transfer exactly; the history
field is inherited by the children. Checking inside the staggered loop, not per load
step, matters for brittle snaps, where the crack crosses the specimen within one step.

**Results.** Against the uniform grid of `../nonlinear/` at the same finest resolution
(coarse elements 1/64, threshold 0.1, buffer 0.05):

| case | mesh | elements | time | peak [kN] |
|---|---|---|---|---|
| tension 256 | uniform | 65536 | 137 s | 0.7799 |
| tension 256 | static band | 11008 | 28 s | 0.7800 |
| tension 256 | adaptive | 7942 to 13021 | 31 s | 0.7804 |
| shear 512 | uniform | 262144 | 3181 s | 0.5428 |
| shear 512 | static quadrant | 94786 | 1387 s | 0.5425 |
| shear 512 | adaptive | 18940 to 52675 | 549 s | 0.5431 |
| shear 512 | adaptive, increment 1e-4 | 18940 to 52456 | 337 s | 0.5431 |

The crack paths coincide. All rows but the last use the load increment 2e-5 of the
uniform driver in shear.

**Where the shear time goes.** Almost all of it is stable crack growth, where every
load step needs one to four hundred staggered iterations; the steps before the peak
cost seconds. About 150 of those iterations per step are the slow linear tail of the
staggered fixed point, paid once per converged step whatever the increment, so 5x
larger increments need 2.3x fewer iterations with the same peak and crack (unlike
tension, where the brittle snap dominates). Beyond 1e-4 the saving is eaten by harder
displacement solves and more remeshing. The tension/compression switch is updated
once per staggered iteration instead of being converged within it, which halves the
displacement solves but not their CG work: that is set by the change in the phase field.

**What did not help in shear.** A monolithic Newton-FGMRES (matrix-free Jacobian, the
couplings rank one per element, block triangular V-cycle preconditioner) taking over
slow load steps converges in one or two Newton steps before the peak but fails during
crack growth even with a backtracking line search, as the history switch, the
compression switch and the advancing crack make the residual too nonsmooth. Refreshing
the V-cycle more often does not help either: even a fresh hierarchy needs 20 to 40 CG
iterations on the broken material (stiffness 1e-7 plus the stiff compression
constraint). A CG tolerance of 1e-5 is 25% faster but shifts the peak by 0.4%.
