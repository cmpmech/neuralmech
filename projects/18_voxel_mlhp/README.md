# High-Order Voxel Finite Elements on the GPU

Linear elasticity and Poisson problems on heterogeneous voxel grids for **Chapter 18
(Simulation Acceleration via GPUs)**, solved with $Q_p$ elements that span
$s \times s (\times s)$ voxels, preintegrated exactly from the voxel data, applied
matrix-free and preconditioned by multigrid. `voxel.py` holds the solver
(`VoxelMultigrid`), `materials.py` resolution-independent heterogeneous materials; both
condense the explorations in `18_mlhp_v0`, `18_mlhp_v2` and `18_mlhp_v3`.

## Drivers

- `solve.py`
  one solve of the cantilever with every choice on the command line (`--dim`,
  `--material`, `--resolution`, `--degree`, `--sub`, `--floor`, `--contrast`,
  `--smoothing`), plots the material and $|u|$
- `topopt.py`
  compliance topology optimization (2D half MBB beam, 3D cantilever with an end line
  load) with the design, filter (radius 2 voxels) and sensitivities per voxel and
  $Q_p$ analysis over $s^D$ voxels; the thresholded design is re-evaluated on the voxel
  reference to expose artificially stiff patterns
- `forward.py`
  cantilever (`--physics elasticity` or `poisson`, `--dim 2` or `3`) on a gyroid
  lattice, random inclusions or a log-normal random field (`--material`), comparing
  $(p, s)$ against the voxel reference $(1, 1)$ in accuracy, memory and time over
  growing resolutions

## Non-obvious technicalities (authored by Claude)

**Exact preintegration as fitted quadrature weights.** The voxel coefficient $c$ is
constant per voxel, and every integrand of a $Q_p$ stiffness matrix (products of
gradient components) lies in $Q_{2p}$. With the Lagrange polynomials $\ell_q$ of the
$(2p+1)^D$ Gauss grid of an element, the weights $w_q = \int_e c\,\ell_q\,dx$ therefore
integrate $K_e = \sum_q w_q B_q^\top C B_q$ exactly, identical to summing preintegrated
voxel matrices $\sum_k c_k K_k$ (Yang et al. 2012). A fine element stores
$(2p+1)^D$ floats instead of $s^D$ voxel values or $n^2$ matrix entries, i.e.
$4\,((2p+1)/s)^D$ bytes per voxel (2.7 for $p = 3$, $s = 8$ in 3D), and computing them
is a sum-factorized contraction of the voxel grid with the 1D integrals
$\int_{\text{voxel}} \ell_q$. Rounding $w_q$ to float32 only perturbs the material:
$B_q$ annihilates rigid motions exactly, unlike rounded element matrices
(`18_mlhp_v3`). This generalizes the six-moment bilinear element of `18_mlhp_v2`. For
$s = 1$ (the reference) the kernel reads the voxel coefficient and uses $(p+1)^D$ Gauss
points per voxel instead.

**Matrix-free operator by sum factorization.** One thread block row per element gathers
$u_e$, interpolates the gradients to the points one direction at a time
(Kronbichler and Kormann), forms the weighted stress and integrates back the same way.
In 3D this took the $(2, 4)$ solve from 0.61 to 0.19 s and $(3, 8)$ from 1.8 to 0.28 s
against a point-by-point basis loop. Elements are processed in $2^D$ colors that share
no nodes, so the scatter needs no atomics and sums in a fixed order.

**Hierarchy from the same weights.** $Q_p$ goes to $Q_1$ on the same mesh, then the
mesh coarsens by 2, or by 3 or 5 where 2 does not divide it (a grid ending in 9
elements would otherwise stop at a 5700 dof dense inverse, 0.25 s per design update in
3D). Every coarse level is again matrix-free with $3^D$ weights per element,
fitted directly from the voxel grid; since the spaces are nested and integrated
exactly, this is the Galerkin operator without any triple product. Transfers are
tensor-product interpolation stencils (Gauss-Lobatto positions for the $p$ step,
midpoints for the $h$ steps); Dirichlet masks are injected, as in `18_mlhp_v2`. The
coarse grid must resolve the gaps of the material: on the 2D gyroid slice a coarsest
grid of about 1000 dofs gives 40 CG iterations for every $(p, s)$ and resolution, 300
dofs give over 200 (the known limit of geometric coarsening in `18_mlhp_v2`), hence
`coarse_dofs = 2000` and a dense inverse.

**Cut elements: an element-wise finite cell $\alpha$.** With $p \ge 2$, void of
stiffness $10^{-6}$ inside an element produces polynomial modes with almost no energy
on the solid part (the finite cell ill-conditioning, Jomo et al. 2021). Chebyshev-Jacobi
then needs 200 to over 1000 CG iterations on the gyroid, whatever the smoothing degree.
Before multigrid sees the material, every voxel of an element is therefore raised to
$\alpha = 10^{-3}$ times the stiffest voxel of the same element (`floor`). Only elements
that hold solid and void change, void away from the solid keeps its stiffness: a local
version of the global $\alpha$ of the finite cell method. The contrast inside an element
is then at most $10^3$, and plain Jacobi smoothing gives resolution independent
iteration counts ($(2, 4)$: 36 to 51, $(3, 8)$: 41 to 77 on the 3D gyroid from 32 to
160 voxels per length), also for void of $10^{-9}$. The price is a slightly stiffer
body, a coat of $\alpha$ around the solid at most one element thick. On the 3D gyroid
it lowers the compliance by 0.3% for $(2, 4)$ and 1.4% for $(3, 8)$ at 128 voxels per
length, 0.3% and 0.7% at 160, and 0.1% or less in 2D at 4096 voxels per length; it
shrinks with the resolution like the discretization error itself (about a fifth and a
third of it). $\alpha = 10^{-4}$ costs a quarter of that but twice the iterations for
$(3, 8)$, growing with the resolution.

**What did not work for cut elements.** (i) Element-block additive Schwarz on the cut
elements restores the iterations of the voxel reference, but $n^2$ floats per element
cost 290 to 1500 bytes per voxel in 3D. (ii) Low-rank corrections $U U^\top$ of Jacobi
from the eigenmodes of the Jacobi-scaled cut element blocks below 0.1 (the previous
version of this solver) cost 34 of 49 bytes per voxel and 4 s of setup for $(3, 8)$ in
3D, still needed 162 iterations and did not converge at void $10^{-9}$, where the
float32 weights cannot resolve the modes. (iii) Flooring only the multigrid hierarchy
and running CG on the true operator: 260 to over 1000 iterations, the tiny modes stay
in the operator. (iv) A low-order refined preconditioner, $Q_1$ on the node submesh of
$Q_p$ (for $p = 2$ exactly the $(1, 2)$ operator): over 1000 iterations, a polynomial
with little energy on the solid part is not a low energy piecewise linear function, so
both are not spectrally equivalent under void. (v) Nodal $3 \times 3$ block Jacobi: 5
to 10% fewer iterations than Jacobi for 3 more bytes per voxel. (vi) Raising each
element by $\alpha$ times its mean instead of its maximum (differentiable, for topology
optimization) needs about twice the iterations at the same accuracy.

**Sensitivities per voxel.** The compliance gradient with respect to a voxel
coefficient is minus the strain energy of that voxel, $u_e^\top K_v u_e$, which a kernel
evaluates at $(p + 1)^D$ Gauss points per voxel (exact for $Q_p$) straight from the
element displacements, after removing the element translation in float64 (as in
`18_mlhp_v3`). The floor couples the voxels of an element, so `energy_gradient` adds
its chain rule; with a plain maximum as the floor, every floored voxel would get zero
sensitivity and void next to a member could not grow back. The smooth 8-norm keeps the
iterations of the maximum (3D gyroid $(2, 4)$: 56 against 51, $(3, 8)$: 84 against 77)
and its gradient agrees with finite differences to $2 \cdot 10^{-5}$.

**Multi-resolution designs.** On the 2D half MBB beam (480 x 120 voxels) and the 3D
cantilever (64 x 32 x 32), $(2, 4)$ gives the designs of the voxel reference: the
thresholded designs evaluated on $(1, 1)$ have the same compliance (2D 387.8 against
388.0 with the density filter, 3D 224.6 against 226.8). $(3, 8)$ violates the rule
$p \ge \mathrm{round}(0.75\, s)$ of Groen et al. (2017) far more and fills whole regions
with element-aligned stripes under the density filter (1438 on the voxel reference);
the sensitivity filter smooths them out but leaves dashed members. The floor is not
the cause: $(2, 4)$ shows the same dashed member with $\alpha = 0$, at three times the
CG iterations.

**The coarse inverse per design update.** Inverting the dense coarse matrix (about 2000
dofs) in float64 took 175 to 350 ms on this GPU, ten times a whole solve at 480 x 120
voxels. The float32 inverse of the Jacobi-scaled matrix gives the same CG iterations
down to void $10^{-9}$ in 16 ms, and assembling it on the GPU took a design iteration
from 0.25 to 0.05 s.

**Topology optimization timings.** Density filter, radius 2 voxels, 300 design
iterations, compliance of the thresholded design on the voxel reference; seconds per
design iteration on the laptop GPU. Groen marks $p \ge \mathrm{round}(0.75\, s)$.

| $(p, s)$ | Groen | 2D 1920 x 480: s / iter | compliance | 3D 144 x 72 x 72: s / iter | compliance |
|---|---|---|---|---|---|
| $(1, 1)$ reference | | 0.41 | 406.5 | 1.05 | 215.3 |
| $(1, 2)$ | no | 0.18 | 406.3 | 0.48 | 215.8 |
| $(2, 4)$ | no | 0.19 | 405.9 | 0.44 | 215.3 |
| $(2, 3)$ | yes | 0.24 | 406.4 | 0.37 | 215.5 |
| $(3, 4)$ | yes | 0.24 | 406.3 | 0.82 | 215.5 |
| $(4, 6)$ | yes | 0.28 | 406.2 | 0.85 | 215.6 |
| $(6, 8)$ | yes | 1.62 | 406.0 | | |
| $(3, 8)$ | no | | | 0.49 | 225.3 |

All designs but $(3, 8)$ match the reference, so $(2, 4)$ is the default and $(1, 2)$ the
fallback. The Groen settings have 2 to 3 times the dofs of $(2, 4)$, and Jacobi on the
$Q_p$ level weakens for $p \ge 4$ (single CG solves up to 1000 iterations; $(6, 8)$
would need intermediate $p$ levels). The 3D times were measured before the coarsening
by 3, which took the $(2, 4)$ update from 286 to 28 ms and its solve to 27 ms against
265 ms for the reference. 
**Memory of a design iteration.** The solver holds about 40 bytes per voxel for
$(2, 4)$, the driver's design fields about 25 more, but the pool peaked at 160 (2D) to
185 (3D) bytes per voxel: the floor and fitting every level from the voxel grid made
float64 temporaries of the whole grid in every design update. The floor is now one
kernel per element into two persistent float32 grids (the raised coefficients and the
derivative factor for the sensitivities), the levels up to $8^D$ voxels per element
are fitted by a kernel straight into their weights, and coarser levels are contracted
exactly from the weights of the next finer level ($w_Q = \sum_{c, q} w_{c, q}\,
\ell_Q(x_{c, q})$, exact since the child weights integrate the parent polynomials).
This brought the peak to 88 (3D) to 98 (2D) bytes per voxel for $(2, 4)$ and 94 to 101
for $(1, 2)$, at the same CG iterations; the fitted weights change by float32 rounding
only. The 6 GB laptop GPU then fits about 50 million voxels in 2D or 3D, a 32 GB GPU
about 300 million.

**Results.** Cantilever with 1e-6 void (gyroid), 1e3 stiff inclusions or a log-normal
field spanning 1e3, against the voxel reference $(1, 1)$ at the same resolution; laptop
RTX PRO 500 (6 GB). Bytes per voxel of the solver (vectors, weights, coarse inverse),
energy error $(f^\top u - f^\top u_{\text{ref}})/f^\top u_{\text{ref}}$, $\alpha = 10^{-3}$.

2D elasticity, 4096 x 2048 voxels:

| material | $(p, s)$ | B / voxel | CG | setup | solve | energy error |
|---|---|---|---|---|---|---|
| gyroid | $(1, 1)$ reference | 159 | 41 | 0.5 s | 7.4 s | |
| gyroid | $(1, 2)$ | 49 | 40 | 0.3 s | 2.5 s | -0.39% |
| gyroid | $(2, 4)$ | 46 | 46 | 0.3 s | 2.4 s | -0.30% |
| gyroid | $(3, 8)$ | 23 | 65 | 0.2 s | 1.4 s | -0.54% |
| inclusions | $(1, 1)$ reference | 159 | 41 | 0.5 s | 7.4 s | |
| inclusions | $(1, 2)$ | 49 | 33 | 0.3 s | 2.1 s | -0.17% |
| inclusions | $(3, 8)$ | 23 | 65 | 0.2 s | 1.4 s | -0.31% |
| random | $(1, 1)$ reference | 159 | 16 | 0.5 s | 3.0 s | |
| random | $(1, 2)$ | 49 | 15 | 0.3 s | 1.0 s | -2e-5 |
| random | $(3, 8)$ | 23 | 16 | 0.2 s | 0.4 s | -4e-6 |

3D elasticity, 256 x 128 x 128 voxels:

| material | $(p, s)$ | B / voxel | CG | setup | solve | energy error |
|---|---|---|---|---|---|---|
| gyroid | $(1, 1)$ reference | 220 | 31 | 0.7 s | 9.8 s | |
| gyroid | $(1, 2)$ | 42 | 33 | 0.3 s | 1.9 s | -3.7% |
| gyroid | $(2, 4)$ | 36 | 51 | 0.2 s | 1.7 s | -2.8% |
| gyroid | $(3, 8)$ | 14 | 77 | 0.1 s | 0.8 s | -4.9% |
| random | $(1, 1)$ reference | 220 | 18 | 0.7 s | 5.8 s | |
| random | $(1, 2)$ | 42 | 18 | 0.2 s | 1.1 s | -0.58% |
| random | $(2, 4)$ | 36 | 19 | 0.2 s | 0.7 s | -0.09% |
| random | $(3, 8)$ | 14 | 23 | 0.1 s | 0.2 s | -0.09% |

For smooth heterogeneity high order pays twice: $(3, 8)$ is more accurate than bilinear
or trilinear $2^D$ elements at a third to a half of their memory and several times
faster. With sharp interfaces the error is set by the kink across the interface, which
no polynomial inside an element captures, so $(2, 4)$ and $(3, 8)$ are about as
accurate as $(1, 2)$ at the same number of dofs, plus the $\alpha$ coat; all errors
shrink with resolution (3D gyroid $(3, 8)$: -11.6%, -4.9%, -2.6% at 64, 128, 160 voxels
per length). $(2, 4)$ is the robust default: at most the error of $(1, 2)$ for less
memory and time. $(p, s)$ with $p / s \ge 3/4$ ($(2, 2)$, $(3, 4)$) have more dofs than
the reference or close to it and never pay off.
