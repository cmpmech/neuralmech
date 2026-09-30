# Condensed High-Order Voxel Elements on the GPU

Third attempt at fast voxel topology optimization for **Chapter 18 (Simulation
Acceleration via GPUs)**, after `18_mlhp` (direct solvers) and `18_mlhp_v2` (multigrid,
bilinear elements). Here each element spans $s \times s$ voxels with a full tensor
$Q_p$ space, its interior is condensed out, and multigrid-preconditioned CG runs on the
element edges only. The design, filter and sensitivities stay per voxel, so the filter
radius can remain 2 voxels. `highorder.py` holds the solver (`HighOrderMultigrid`).

## Drivers

- `topopt_elasticity2D_highorder.py`
  the half MBB compliance optimization of `18_mlhp_v2/topopt_elasticity2D_structured.py`
  with $p = 2$ elements over $4 \times 4$ voxels and filter radius 2 voxels

## Non-obvious technicalities (authored by Claude)

**Which $(p, s)$ keep the designs clean.** Groen et al. (2017) recommend
$p \ge \mathrm{round}(0.75\, s)$ against artificially stiff patterns for a filter of
radius $2h$. For the conic density filter with $r_{\min} = 2$ voxels here the limit is
somewhat more lenient, but it has to be checked at the target resolution and by eye.
At 480 x 120 voxels every $(p, s)$ up to $(2, 5)$ and $(3, 8)$ looked clean, while
$(1, 3)$, $(2, 6)$ and $(2, 8)$ showed dot patterns (402, 598 and 8805 against 380 on a
one-voxel-per-element reference). At 5120 x 1280 voxels the optimizer grows fans of
thin members in low-stress regions, and there $(2, 5)$ and $(3, 8)$ fill whole areas
with element-aligned stripes one voxel thick (0.14% and 0.08% of the voxels against
0.03% for one voxel per element). The compliance hardly shows this, since those regions
carry little load. $(2, 4)$, $(3, 4)$ and $(3, 5)$ stay clean (below 0.01%).

**Static condensation onto the skeleton.** Each element's interior nodes are
eliminated once per design update, $S_e = K_{bb} - K_{bi} K_{ii}^{-1} K_{ib}$, so CG only
sees the element vertices and edge nodes, about $2 (2p - 1) / s^2$ dofs per voxel
(0.16 for $(3, 8)$ against 0.5 for bilinear $2 \times 2$ and 2 for one voxel per
element). The Gauss-Lobatto nodes keep the skeleton a structured grid: vertices, then
the $p - 1$ nodes of every horizontal and every vertical edge, all found by index
arithmetic. The operator runs element by element (a thread multiplies the packed
upper triangle of $S_e$) into a buffer, and a node kernel sums the two or four element
contributions per node, so there are no atomics. Interior displacements for the
sensitivities come back from the stored $X = K_{ii}^{-1} K_{ib}$.

**Rigid motions and float32.** Rounding $S_e$ to float32 gives each element a spurious
stiffness of order $10^{-7}$ against rigid motion, and in a bending beam the element
displacements are almost entirely rigid: with the stored float32 matrices the
compliance was 10% off. Removing the element translations and rotation from $u_e$ in
float64 before the float32 product (and projecting the result the same way) brings it
to $10^{-8}$, with float32 storage and arithmetic. The same effect is why
`18_mlhp_v2` needs a float64 outer operator and why its six-moment elements annihilate
translations exactly.

**A floor on the void inside elements.** An element that mixes solid and $10^{-9}$ void
loses the void stiffness to float32 rounding (its smallest non-rigid eigenvalue drops
to $10^{-17}$ of the largest for $8 \times 8$ voxels), the void part then floats and CG
stalls. The solver raises voxel coefficients below $10^{-6}$ to $10^{-6}$; the compliance
changes by about $10^{-4}$, and against a direct solver with the same floor the GPU
solver agrees to $10^{-7}$ in compliance and $10^{-6}$ in sensitivities.

**Multigrid.** The skeleton is smoothed with degree-1 Chebyshev on an additive Schwarz
preconditioner whose blocks are the assembled operator restricted to one element's
skeleton dofs (own $S_e$ plus the neighbours sharing its edges and vertices), inverted
by Cholesky in float32 after diagonal scaling. Point Jacobi needs 3 to 7 times more CG
iterations for $p = 3$ (enough for $p = 2$), and blocks without the neighbour
contributions 2.5 times more. The next level is bilinear on the element vertices,
linearly interpolated along each edge, with Galerkin matrices $Q^\top S_e Q$ that must be
accumulated in float64: formed in float32 from the rounded $S_e$ they raised a cold solve
from 43 to 85 iterations. Coarser levels are the bilinear hierarchy of `18_mlhp_v2`.

**Setup cost.** Condensation (float64 Cholesky of the $2(p-1)^2$ interior block per
element) and the Schwarz inverses run one thread per element. Elements of uniform
voxels, and Schwarz blocks with a uniform $3 \times 3$ neighbourhood, are the unit
element scaled; they are written by a separate kernel and the costly kernels only see a
compacted index list of the rest (checking uniformity inside one kernel left most warps
divergent). On the converged 8192 x 2048 design this took the update from 590 to 83 ms.

**Chebyshev bounds lag while members form.** The largest eigenvalues come from 8
warm-started power iterations per design update. In the phase where thin members form
the estimate lagged by 15% and CG needed up to 500 iterations; a safety factor of 1.3
instead of 1.1 cut the mean over the first 100 design iterations from 34.8 to 25.5 and
the maximum to 55.

**Timings.** Half MBB beam, 5120 x 1280 voxels, $r_{\min} = 2$ voxels, laptop RTX PRO
500 (6 GB), seconds per design iteration and bytes per voxel:

| $(p, s)$ | s / iteration | B / voxel | stripes |
|---|---|---|---|
| $(3, 8)$ | 0.19 | 145 | yes |
| $(2, 5)$ | 0.24 | 129 | yes |
| $(2, 4)$ | 0.35 | 206 | no |
| $(3, 5)$ | 0.38 | 251 | no |
| $(1, 2)$, `18_mlhp_v2` | 0.38 | 114 | no |
| $(3, 4)$ | 0.67 | 372 | no |
| $(1, 1)$, `18_mlhp_v2` | 0.92 | 226 | no |

Among the clean settings, $(2, 4)$ gains only 10% over bilinear $2 \times 2$ elements
(0.69 against 0.74 s at 8192 x 2048 voxels, where $(3, 4)$ no longer fits into 6 GB).
The larger speed-ups of $(3, 8)$ come with the stripes. With $p = 3$ on small elements
the condensed matrices grow faster than the dofs shrink, and the operator is bound by
reading them.
