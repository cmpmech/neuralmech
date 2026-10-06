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
tensor-product Lagrange interpolation tables, the $Q_q$ basis at the Gauss-Lobatto
nodes of $Q_p$ for a $p$ step and linear at the fine nodes for an $h$ step; Dirichlet
masks are injected, as in `18_mlhp_v2` (a $Q_q$ node takes the mask of the nearest
$Q_p$ node of its element). The coarse grid must resolve the gaps of the material: on
the 2D gyroid slice a coarsest grid of about 1000 dofs gives 40 CG iterations for every
$(p, s)$ and resolution, 300 dofs give over 200 (the known limit of geometric
coarsening in `18_mlhp_v2`), hence `coarse_dofs = 2000` and a dense inverse.

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
optimization) needs about twice the iterations at the same accuracy. (vii) The
GEVS-like correction (Jacobi plus $v v^\top / \lambda$ for every eigenpair
$K_c v = \lambda D_c v$ of the principal submatrix of an element below a threshold
$\tau$), used inside the Chebyshev smoother of the $Q_p$ level, prototyped with a CPU
assembly: on the 2D gyroid (240 x 120 voxels, CG to $10^{-8}$) it takes $(4, 6)$ from
156 to 28 and $(6, 8)$ from 314 to 60 iterations together with an intermediate $Q_3$
level, and $(4, 6)$ converges in 39 iterations even without the floor ($\alpha = 0$, 3%
softer and closer to the true material). But a fixed $\tau = 0.3$ keeps up to 22 modes
per element at $p = 6$ (135 bytes per voxel), because the Jacobi-scaled patch of an
uncut element already has its lowest eigenvalue at about $0.6 / p$; a threshold
relative to that keeps 4 to 9 floats per voxel but only reaches 52 and 153. On
optimized topology optimization designs it does not help at all ($(4, 6)$: 43 to 42,
$(6, 8)$: 48 to 43 on the 1920 x 480 MBB design), and every design update would need
one dense eigenproblem per cut element, $n = D (p + 1)^D$: batched `eigh` costs 0.1 ms
per element at $n = 50$ (2D, $p = 4$) and 10 ms at $n = 375$ (3D, $p = 4$), seconds
to minutes per update on large 3D grids. It is the right tool for sharp voids in a
forward solve, not for topology optimization.

**Where the iterations of high $p$ come from.** On the smooth log-normal field the CG
iterations grow only mildly with $p$ (2D, 960 x 480: 16 for $(2, 4)$, 20 for $(4, 6)$,
32 for $(6, 8)$); on the gyroid they jump (37, 102, 194), and raising the floor to
$10^{-2}$ halves them, while a larger coarse grid or more Chebyshev steps on the $Q_p$
level do not pay: the cost is cut-element modes of the $Q_p$ level, plus the large step
from $Q_p$ to $Q_1$ for $p \ge 5$. On optimized topology optimization designs both
matter less. At fixed $s = 6$ the iterations rise from 37 ($p = 1$) to 48 ($p = 4$) and
47 ($p = 6$) on the 1920 x 480 MBB design, against 32 for the voxel reference: the floor
of all settings is the geometric $h$ hierarchy below thin members, which no fine level
smoother changes.

**Intermediate $p$ levels.** For $p \ge 5$ the hierarchy goes $Q_p \to Q_{p/2} \to
Q_1$ on the same mesh (`p_levels`), the $Q_{p/2}$ weights fitted from the voxels like
every level, smoothed like the finest level. $(6, 8)$ then needs 106 instead of 194
iterations on the 2D gyroid and 48 instead of 72 on the 1920 x 480 MBB design (solve
0.52 instead of 0.74 s and 0.48 instead of 0.61 s), for 4 to 10 more bytes per voxel.
For $p = 4$ the extra level costs as much as it saves ($(4, 6)$ design: 43 instead of
48 iterations, 0.33 instead of 0.30 s), so it starts at $p = 5$.

**Sensitivities per voxel.** The compliance gradient with respect to a voxel
coefficient is minus the strain energy of that voxel, $u_e^\top K_v u_e$, which a kernel
evaluates at $(p + 1)^D$ Gauss points per voxel (exact for $Q_p$) straight from the
element displacements, after removing the element translation in float64 (as in
`18_mlhp_v3`). The floor couples the voxels of an element, so `energy_gradient` adds
its chain rule; with a plain maximum as the floor, every floored voxel would get zero
sensitivity and void next to a member could not grow back. The smooth 8-norm keeps the
iterations of the maximum (3D gyroid $(2, 4)$: 56 against 51, $(3, 8)$: 84 against 77)
and its gradient agrees with finite differences to $2 \cdot 10^{-5}$.

**A convex floor.** Adding $\alpha$ times the element's 8-norm to every voxel also
raised homogeneous solid elements by $\alpha$, a uniform 0.1% stiffening that was most
of the energy error on smooth fields (2D log-normal field, 1024 x 512: $-1.07 \cdot
10^{-3}$ for $(2, 4)$, $-5.6 \cdot 10^{-5}$ without floor). Every voxel now moves the
fraction $\alpha$ towards the 8-norm instead, $c' = (1 - \alpha)\, c + \alpha\,
\|c\|_8$: void still gains $\alpha \|c\|_8$, a homogeneous element keeps its
coefficient. The CG iterations are the same in every case tested (2D and 3D gyroid,
optimized designs, $p$ up to 6); the energy error drops by $10^{-3}$ everywhere, to
$-6.7 \cdot 10^{-5}$ on the log-normal field and on the 1920 x 480 MBB design to within
$5 \cdot 10^{-5}$ of the unfloored operator ($(2, 3)$: $+1.1 \cdot 10^{-5}$ against
$-9.9 \cdot 10^{-4}$ before), at 1 to 35 times fewer CG iterations than without the
floor. On the gyroid what remains is the coat of $\alpha$ in the void of cut elements
($(3, 4)$: $-3.2 \cdot 10^{-3}$ against $-4.2 \cdot 10^{-3}$ before and $-2.5 \cdot
10^{-4}$ unfloored). The gradient, $(1 - \alpha)$ times the voxel energy plus
$\alpha\, \partial \|c\|_8 / \partial c$ times the element energy, agrees with
central differences along random directions to $10^{-3}$, the float32 noise level, as
the additive one did.

**Multi-resolution designs.** On the 2D half MBB beam (480 x 120 voxels) and the 3D
cantilever (64 x 32 x 32), $(2, 4)$ gives the designs of the voxel reference: the
thresholded designs evaluated on $(1, 1)$ have the same compliance (2D 387.8 against
388.0 with the density filter, 3D 224.6 against 226.8). $(3, 8)$ violates the rule
$p \ge \mathrm{round}(0.75\, s)$ of Groen et al. (2017) far more and fills whole regions
with element-aligned stripes under the density filter (1438 on the voxel reference);
the sensitivity filter smooths them out but leaves dashed members. The floor is not
the cause: $(2, 4)$ shows the same dashed member with $\alpha = 0$, at three times the
CG iterations.

**Artificially stiff patterns under a distributed load.** The MBB beam hardly
discriminates: at 480 x 120 every $(p, s)$ up to $s = 8$ lands within 0.3% of the
reference on $(1, 1)$, except $(3, 8)$. Groen's critical case does, a 2 x 1 cantilever
clamped on the left with a distributed load on the top edge (volume 0.4, a passive
solid layer of 2 voxels under the load), where the low-load region below the top edge
invites thin horizontal stripes that the coarse analysis overrates. Thresholded
designs on $(1, 1)$ after 300 iterations, density filter of radius 2 voxels; the
artifact-free $(2, 1)$, $(2, 2)$ and $(3, 3)$ give 13.92, 13.85 and 13.84 at 480 x 240,
the size of the noise:

| $(p, s)$ | Groen | 480 x 240 | 960 x 480 | 480 x 240, sensitivity filter |
|---|---|---|---|---|
| $(1, 1)$ reference | | 13.85 | 14.10 | 13.52 |
| $(2, 3)$ | yes | 15.28 | 14.13 | 13.50 |
| $(3, 4)$ | yes | 14.64 | 14.10 | 13.52 |
| $(4, 5)$ | yes | 15.62 | 20.89 | |
| $(5, 6)$ | yes | 14.85 | 14.10 | |
| $(6, 8)$ | yes | 15.01 | 20.89 | 16.80 |
| $(4, 6)$ | no | 24.76 | | 13.58 |
| $(3, 5)$ | no | 25.13 | | |
| $(5, 8)$ | no | 40.74 | | |
| $(1, 2)$ | no | 381.6 | | 14.94 |
| $(2, 4)$ | no | 573.8 | 41.73 | 15.13 |
| $(3, 6)$ | no | 1317 | 184.1 | |
| $(4, 8)$ | no | 1796 | | |
| $(3, 8)$ | no | 31045 | | |

Every setting that breaks Groen's rule (with 4.5 rounded up, $(4, 6)$ breaks it) shows
the stripes, $(2, 4)$ and $(1, 2)$ included, with 1.8 to 2000 times the compliance
after thresholding. The rule is necessary but not sufficient at radius 2: $(4, 5)$ and
$(6, 8)$ fail at 960 x 480. At 480 x 240 with the density filter every setting with
$s \ge 3$ ends 6 to 13% above the reference with faint stripes under the top edge,
Groen's settings included; from 960 x 480 on, and with the sensitivity filter, $(2, 3)$
and $(3, 4)$ are clean, i.e. elements of at most 4 voxels with $p \ge 0.75\, s$; the
element size in voxels matters more than $p / s$. With $\alpha = 0$ the designs are
the same ($(2, 4)$: 491, $(2, 3)$: 15.27, $(3, 4)$: 14.73, $(6, 8)$: 14.85 at
480 x 240) at 1.4 to 36 times the CG iterations, so the floor plays no part in the
stripes, and the convex floor above leaves them as they are ($(2, 3)$: 15.28,
$(3, 4)$: 14.64, $(2, 4)$: 576.7; 960 x 480 $(2, 3)$: 14.21). The
stripes grow with the resolution: at 1920 x 960 the reference gives 14.44, $(2, 3)$
and $(3, 4)$ 14.43 and $(2, 4)$ 367.6 with 21 disconnected pieces. The same
cantilever in 3D (2 x 1 x 0.5, 144 x 72 x 36 voxels, volume 0.15, load on the top
face) agrees: reference 76.17, $(2, 2)$ 76.13, $(2, 3)$ 75.97, $(3, 4)$ 75.96, $(1, 2)$
78.10 and $(2, 4)$ 114.2, at 0.60, 0.62, 0.20, 0.28, 0.15 and 0.11 s per iteration.
On the 3D cantilever with an end line load (144 x 72 x 72), in contrast, $(2, 4)$,
$(2, 3)$, $(3, 4)$ and $(1, 2)$ all stay within 0.3% of the reference (215.3), like
the MBB beam. $(2, 4)$ remains the cheapest choice where every region carries load;
where low-load regions matter, $(2, 3)$ is the cheapest safe one, 1.6 to 1.8 times the
time of $(2, 4)$ and a third to two thirds of the reference.

**The coarse inverse per design update.** Inverting the dense coarse matrix (about 2000
dofs) in float64 took 175 to 350 ms on this GPU, ten times a whole solve at 480 x 120
voxels. The float32 inverse of the Jacobi-scaled matrix gives the same CG iterations
down to void $10^{-9}$ in 16 ms, and assembling it on the GPU took a design iteration
from 0.25 to 0.05 s. It is stored in float32 too and applied with one warp per row
(coalesced reads), 0.4 ms per V-cycle with one thread per float64 row before.

**A W-cycle on the coarse meshes.** On optimized designs the geometric $h$ hierarchy
sets the iteration count of every $(p, s)$ (above). Visiting every coarsened mesh
twice, the second time on the residual of the first (`w_cycle`, `--cycle W`), halves
the CG iterations on the 1920 x 480 MBB design for $(1, 1)$ (32 to 18) and $(2, 3)$
(36 to 18), but not on the gyroid, where the $Q_p$ level limits. On the launch-bound
2D grids of this GPU it does not pay ($(1, 1)$: 0.64 to 0.74 s per solve); in 3D,
where the coarse meshes cost $2^{-3}$ per level, it does on topology optimization
designs (the 3D cantilever design at 288 x 144 x 72: $(2, 4)$ 30 to 24 iterations,
0.78 to 0.60 s, $(1, 2)$ 29 to 19, 1.19 to 0.95 s), except for $(2, 3)$, whose
iterations stay at 33 (2.1 to 2.3 s). It starts at the first coarsened mesh, the
$Q_1$ level on the fine mesh is visited once. Over whole 3D optimizations
(144 x 72 x 72, 300 iterations, interleaved twice, the same designs to five digits)
it is never slower: mean CG 25 to 21, 23 to 14 and 18 to 15 and time per design
iteration 0 to 9%, 14 to 18% and 7 to 9% lower for $(2, 4)$, $(1, 2)$ and $(2, 3)$;
on large grids of a large GPU the coarse levels weigh even less. `topopt.py` therefore
uses the W-cycle in 3D and the V-cycle in 2D (`--cycle` overrides), the solver keeps
the V-cycle as its default.

**Chebyshev bounds by Lanczos.** The largest eigenvalue of $D^{-1} A$ on every level
took 24 power iterations per design update, the largest part of the setup (24 to 64 ms
of 46 to 89 ms at 1920 x 480). Twelve Lanczos steps in the $D$ inner product from the
same fixed random start give the same CG iterations at a third less cost; the Ritz value
converges to the extreme eigenvalue much faster than the power iteration.

**Topology optimization timings.** Density filter, radius 2 voxels, 300 design
iterations, compliance of the thresholded design on the voxel reference; seconds per
design iteration on the laptop GPU, which ran thermally throttled (SM clock 1680 of
3090 MHz), so the reference column is the yardstick. Groen marks
$p \ge \mathrm{round}(0.75\, s)$, 4.5 rounded up.

| $(p, s)$ | Groen | 2D 1920 x 480: s / iter | compliance | 3D 144 x 72 x 72: s / iter | compliance |
|---|---|---|---|---|---|
| $(1, 1)$ reference | | 0.49 | 406.5 | 1.61 | 215.3 |
| $(1, 2)$ | no | 0.21 | 406.3 | 0.37 | 215.8 |
| $(2, 4)$ | no | 0.18 | 405.9 | 0.27 | 215.3 |
| $(2, 3)$ | yes | 0.30 | 406.5 | 0.48 | 215.5 |
| $(3, 4)$ | yes | 0.35 | 405.9 | 0.70 | 215.5 |
| $(4, 6)$ | no | 0.35 | 406.2 | 1.03 | 215.6 |
| $(5, 6)$ | yes | 0.48 | 406.0 | | |
| $(6, 8)$ | yes | 0.48 | 406.0 | | |

On these two problems all designs match the reference (the distributed load above is
where they part). Interleaved runs of the previous and the current solver and driver
(100 design iterations each, alternating, twice) give: $(6, 8)$ 0.69 to 0.53 s per
iteration (mean CG 51 to 28, the intermediate $Q_3$ level), 3D $(2, 4)$ and $(2, 3)$ 2
to 4% faster, 2D $(2, 4)$ and $(2, 3)$ equal within the noise, and 10 to 18% less pool
peak everywhere. Per CG iteration the current solver is 0 to 10% faster and per
update 14 to 23% (Lanczos, float32 coarse inverse); the more accurate Lanczos bound
costs 3 to 5% more CG iterations over a whole optimization, which a safety factor
below 1.1 would recover at some risk. The Groen settings have 1.8 to 2.8 times the
dofs of $(2, 4)$ in 2D and 2.4 to 3.4 times in 3D; $(2, 3)$ is the cheapest of them,
1.6 to 1.8 times the time of $(2, 4)$ and 0.3 to 0.6 times that of the reference.

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

**Less memory per voxel.** Measured exactly (an allocator hook recording the live
bytes, slope between 96 x 48 x 48 and 192 x 96 x 96 voxels of the 3D cantilever), the
peak of a design iteration sat in the sensitivity and update step at 79 bytes per
voxel for $(2, 4)$ and 127 for $(2, 3)$. Four changes bring it to 62 and 104 at the
same designs (compliances equal to five digits): (i) the floor keeps one float per
element (the 8-norm of its coefficients) instead of the two float32 voxel grids, the
fitting kernel applies it on the fly and `energy_gradient` recomputes the derivative
factor from the coefficients it was given (fitted weights bitwise unchanged, gradients
equal to $10^{-6}$); (ii) the $Q_p$ level reads the float64 CG residual directly
instead of a float32 copy and uses $A p$ as its scratch vector, dead once $x$ and $r$
are updated, 8 bytes per dof; (iii) Lanczos needs no stored start vector per level and the
coarse inverse is kept in float32; (iv) `topopt.py` fuses the chain rule of SIMP, the
optimality criterion base and the volume of every bisection step into single kernels
(a reduction instead of a trial design per step), no design-sized temporaries. At
scale the slope holds: 3D $(2, 4)$ peaks at 62 live bytes per voxel at 33.5 and 65.5
million voxels, the memory pool holds 67 with its cached blocks, and 76 million voxels
(672 x 336 x 336) now run on the 6 GB laptop GPU, against about 50 million before.
A 32 GB GPU that held about 337 million voxels at radius 2 should hold about 450
million for $(2, 4)$ and about 280 million for $(2, 3)$ (109 bytes per voxel). The final
check of `topopt.py` on the voxel reference $(1, 1)$ needs about 180 bytes per voxel,
three times the optimization; on the largest grids it has to be skipped or done on a
cropped or coarsened design.

**Results.** Cantilever with 1e-6 void (gyroid), 1e3 stiff inclusions or a log-normal
field spanning 1e3, against the voxel reference $(1, 1)$ at the same resolution; laptop
RTX PRO 500 (6 GB, thermally throttled, so the times are 10 to 30% above an unthrottled
run). Bytes per voxel of the solver (vectors, weights, coarse inverse), energy error
$(f^\top u - f^\top u_{\text{ref}})/f^\top u_{\text{ref}}$, convex floor
$\alpha = 10^{-3}$.

2D elasticity, 4096 x 2048 voxels:

| material | $(p, s)$ | B / voxel | CG | setup | solve | energy error |
|---|---|---|---|---|---|---|
| gyroid | $(1, 1)$ reference | 131 | 42 | 0.6 s | 8.4 s | |
| gyroid | $(1, 2)$ | 42 | 42 | 0.3 s | 3.0 s | -0.39% |
| gyroid | $(2, 4)$ | 39 | 48 | 0.2 s | 2.8 s | -0.29% |
| gyroid | $(3, 4)$ | 76 | 71 | 0.4 s | 7.0 s | -0.05% |
| gyroid | $(3, 8)$ | 19 | 74 | 0.1 s | 1.7 s | -0.53% |
| inclusions | $(1, 1)$ reference | 131 | 41 | 0.6 s | 8.5 s | |
| inclusions | $(1, 2)$ | 42 | 33 | 0.3 s | 2.4 s | -0.17% |
| inclusions | $(2, 4)$ | 39 | 30 | 0.2 s | 2.0 s | -0.25% |
| inclusions | $(3, 4)$ | 76 | 46 | 0.4 s | 4.9 s | -0.15% |
| inclusions | $(3, 8)$ | 19 | 45 | 0.1 s | 1.2 s | -0.35% |
| random | $(1, 1)$ reference | 131 | 16 | 0.6 s | 3.5 s | |
| random | $(1, 2)$ | 42 | 15 | 0.3 s | 1.3 s | -2.1e-5 |
| random | $(2, 4)$ | 39 | 15 | 0.2 s | 1.0 s | -4.6e-6 |
| random | $(3, 4)$ | 76 | 16 | 0.4 s | 1.8 s | -3.5e-6 |
| random | $(3, 8)$ | 19 | 16 | 0.1 s | 0.4 s | -6.7e-6 |

3D elasticity, 256 x 128 x 128 voxels:

| material | $(p, s)$ | B / voxel | CG | setup | solve | energy error |
|---|---|---|---|---|---|---|
| gyroid | $(1, 1)$ reference | 181 | 31 | 1.1 s | 13.3 s | |
| gyroid | $(1, 2)$ | 37 | 33 | 0.3 s | 2.6 s | -3.7% |
| gyroid | $(2, 4)$ | 31 | 56 | 0.2 s | 2.7 s | -2.7% |
| gyroid | $(3, 4)$ | 89 | 88 | 0.6 s | 9.9 s | -1.2% |
| gyroid | $(3, 8)$ | 12 | 84 | 0.1 s | 1.1 s | -4.5% |
| random | $(1, 1)$ reference | 181 | 18 | 1.1 s | 7.7 s | |
| random | $(1, 2)$ | 37 | 18 | 0.3 s | 1.4 s | -0.58% |
| random | $(2, 4)$ | 31 | 19 | 0.2 s | 0.9 s | -0.11% |
| random | $(3, 4)$ | 89 | 20 | 0.6 s | 2.3 s | -0.07% |
| random | $(3, 8)$ | 12 | 23 | 0.1 s | 0.3 s | -0.22% |

For smooth heterogeneity high order pays twice: $(3, 8)$ is more accurate than bilinear
or trilinear $2^D$ elements at a third to a half of their memory and several times
faster. With sharp interfaces the error is set by the kink across the interface, which
no polynomial inside an element captures, so $(2, 4)$ and $(3, 8)$ are about as
accurate as $(1, 2)$ at the same number of dofs, plus the $\alpha$ coat; all errors
shrink with resolution (3D gyroid $(3, 8)$: -30.6%, -10.7%, -4.5% at 32, 64, 128 voxels
per length). $(2, 4)$ is the robust default for the forward problem: at most the error
of $(1, 2)$ for less memory and time. $(3, 4)$, Groen's setting with elements of 4
voxels, has 2.25 times the dofs of $(2, 4)$ and costs 60 to 85% of the reference's
time on cut material (the $Q_3$ cut modes), for 2 to 6 times
smaller errors than $(2, 4)$ on the gyroid; for topology optimization it and $(2, 3)$
are what keeps low-load regions free of stripes (above).
