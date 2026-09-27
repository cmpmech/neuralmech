# 8_discretization_research

Research on subvoxel (finite cell style) topology optimization: can a filter let the
analysis run on few, low-order elements with many design voxels each, without the
optimizer exploiting the coarse analysis? All drivers minimize the compliance of a
clamped 2:1 cantilever (traction patch on the right edge) and score every design by its
_true_ compliance, a voxel-level re-analysis ($S = 1$, $p = 2$) of the thresholded design.

- `subvoxel_artifacts.py` -- the classic 1.5-voxel filter for $S \times S$ subvoxel
  elements of degree $p$; reproduces the artifacts and their fake stiffness
- `closing_filter.py` -- the proposed filter: 1.5-voxel filter for the solid, plus a
  morphological closing of radius $R = S/p$ for the void, against fine references
- `mbb_closing_filter.py` -- `closing_filter.py` on the half MBB beam (3:1, load and
  roller on 16-voxel patches)
- `multiscale_element.py` -- multiscale elements (condensed voxel interior, polynomial
  edges), to show that a better element alone does not stop the exploitation

## Findings (authored by Claude)

**Mechanism.** The element stiffness is $\sum_s E_s K_s$, the Voigt mean of the voxel
moduli seen through a degree-$p$ displacement field. SIMP makes grey material
uneconomical ($\rho = 0.5$ gives $E = 0.125$), but 0/1 voxel patterns inside an element
are credited close to the Voigt mean $E \approx 0.5$ whether or not they connect. The
optimizer therefore builds fake sub-element microstructures: islands, laminae with
1-2 voxel gaps, slits. Every coarse space is a subspace of the voxel space, so the coarse
compliance is always a lower bound: the fake stiffness is systematic, not noise.

**It is the void, not the solid.** Thin solid members are harmless: a 2-voxel strut
loaded axially is a Voigt laminate, and the coarse field gets its stiffness right. What
the coarse field cannot see are void features narrower than its node spacing $S/p$: it
bridges them. Hence the proposed filter decouples the two length scales:

$$\rho = \mathcal{P}\Big(\mathcal{E}_R\big(\mathcal{D}_R(\mathcal{P}(F_{1.5}\,x))\big)\Big), \qquad R = S/p,$$

a classic density filter $F_{1.5}$ with projection $\mathcal{P}$ (solid length scale
1.5 voxels), then a smooth closing, dilation $\mathcal{D}_R$ then erosion
$\mathcal{E}_R$ over a flat disc of radius $R$ voxels, that fills every void narrower
than about $2R$. A closing only adds material, so the 1.5-voxel solid minimum is untouched
(checked: $\le 0.2\%$ of the solid is thinner than 2 voxels, as in the fine designs under
the same filter; the classic fine design has 2.3%).

**Results.** `closing_filter.py`, 480 x 240 voxels, volume fraction 0.15 (thin members),
250 MMA iterations with projection continuation. The ratio column is true / model
compliance; for clean designs the coarse discretization error alone gives about 1.02 to
1.035.

| run | dofs | true compliance | ratio | filter / FEM time |
|---|---|---|---|---|
| fine $S=1$, classic | 232k | 8.11 | 1.022 | 2 s / 59 s |
| fine $S=1$, $R=4$ | 232k | 8.39 | 1.015 | 10 s / 61 s |
| $S=8$, $p=2$, $R=4$ | 14.8k | 8.47 | 1.031 | 10 s / 6 s |
| $S=16$, $p=4$, $R=4$ | 14.8k | 8.49 | 1.036 | 9 s / 16 s |
| fine $S=1$, $R=8$ | 232k | 8.66 | 1.014 | 11 s / 59 s |
| $S=16$, $p=2$, $R=8$ | 3.8k | 8.91 | 1.049 | 9 s / 4 s |
| $S=8$, $p=1$, $R=8$ | 3.8k | 9.08 | 1.067 | 9 s / 2 s |

Without the closing, the same coarse discretizations give ratios of $10^6$ to $10^7$
(`subvoxel_artifacts.py`). With it, the coarse designs keep the topology and the thin
members of the fine reference under the same filter, at 1 to 5% higher true compliance,
with 16 to 60x fewer dofs and a 4 to 30x cheaper analysis. The closing itself costs 3.4%
($R = 4$) and 6.7% ($R = 8$) in the fine reference, against the classic filter alone. Run-to-run
differences of a few percent are local minima: small setting changes (e.g., the dilation
beta) moved single runs by up to 4% in either direction.

**MBB beam: only node spacings up to 4 voxels hold.** `mbb_closing_filter.py`, 480 x 160
voxels, same settings:

| run | dofs | true compliance | ratio |
|---|---|---|---|
| fine $S=1$, $R=4$ | 155k | 5.15 | 1.017 |
| $S=8$, $p=2$, $R=4$ | 9.9k | 5.31 | 1.053 |
| $S=16$, $p=4$, $R=4$ | 9.9k | 5.23 | 1.036 |
| fine $S=1$, $R=8$ | 155k | 5.51 | 1.018 |
| $S=16$, $p=2$, $R=8$ | 2.6k | 11.29 | 2.17 |
| $S=8$, $p=1$, $R=8$ | 2.6k | 11.60 | 2.24 |

At node spacing 8 the thin diagonal strut comes out dashed: gaps of 5 to 7 voxels across
a 1.5 to 2 voxel strut. Such a gap is a void neck between two large voids, which a closing
only fills up to a width of about $2\sqrt{R w}$ for a strut of width $w$ (4 voxels on the
digital disc, for any $R$). The coarse field bridges it, so the strut is fake. The
cantilever happened not to grow such a strut; the MBB, with its long slender diagonals,
does. The dependable range is therefore $S/p \le 4$ (still 15x fewer dofs).

**The radius rule is tight.** Radii below $S/p$ fail: $S=8, p=2, R=2$ gives a ratio of
1.6, and $S=16, p=2, R=4$ gives 5.2 to $3.6 \cdot 10^6$ (two load cases). $R = 0.75\,S/p$
is marginal (1.04 to 1.07). Only the node spacing matters: $(S, p) = (4, 1), (8, 2),
(16, 4), (24, 6)$ behave alike at the same $R$. At a fixed node spacing, i.e. fixed void
length scale and fixed dof count, the low-order choice is the cheapest to assemble (the
$p = 2$ FEM time is less than half that of $p = 4$ above).

**Remaining exploitation.** Two mild effects remain: voxel-scale bumps and beads on member
surfaces (solid features the field cannot resolve, which cost volume but earn fake
stiffness), and round holes exactly $2R$ wide at junctions. Removing the bumps would need
an opening, which breaks the 1.5-voxel solid minimum, so they are left.

**Multiscale elements are no way out.** `multiscale_element.py` condenses the voxel-Q1
interior of every element, so the element sees how its voxels connect. The optimizer then
moves the fake stiffness onto the element edges, where the traces stay polynomial: sawtooth
island arrays pinned to the edges for $p = 1$ (ratio 24), dashed voxel lines on the edges
for $p = 2$. The per-element condensation costs $O(S^4)$ per voxel, more than a fine solve
in 2D. It also shows why a learned correction of the element stiffness was not pursued:
whatever part of the fine physics the correction leaves out, the optimizer finds and
exploits. Restricting the design space is robust by construction.

## Non-obvious technicalities (authored by Claude)

The benchmark is a cantilever rather than the MBB beam: point loads and point supports are
singular, so the MBB compliance diverges under refinement, and coarse and fine analyses
cannot be compared. The traction patch and its passive solid layer (2 voxels deep) are
aligned to elements for every $S$ dividing 24. The passive layer must enter the design map
**before** the closing: applied afterwards, the optimizer cuts a 1-voxel gap between the
patch and the structure, the patch floats in the fine model, and the true compliance
explodes. Without the layer, the optimizer leaves void voxels under the traction, which the
coarse field spreads onto neighboring solid.

The closing uses Sigmund's log-sum-exp morphology,
$\mathcal{D}_R(\rho)_i = \beta^{-1} \log\big(\sum_j w_{ij} e^{\beta \rho_j} / \sum_j w_{ij}\big)$
with a flat digital disc $w$ of radius $R + 1/2$ and $\mathcal{E}_R(\rho) = 1 - \mathcal{D}_R(1 - \rho)$. Filter plus
Heaviside projection at thresholds $\eta$ and $1 - \eta$ is not a closing: a thin strut
dilates less than a half-plane, so struts either vanish or get fattened depending on
$\eta$. The domain is mirrored at its boundary, so the outside is neither solid nor void
(zero padding would keep everything near the boundary dilated). The disc radius
$R + 1/2$ blunts the one-pixel tips of the digital disc $r^2 \le R^2$: such a tip slips into
a 1-voxel gap across a thin strut from either side, so even the exact closing keeps every
such gap open. The convolution runs by
FFT, so its cost does not depend on $R$; the FFT round-off relative to $e^{\beta}$ limits
$\beta$ to about 20, which is sharp enough ($\log(f)/\beta$ shifts a strut covering the
fraction $f$ of the disc by only a few percent).

The optimizer is MMA. Optimality criteria bisect the volume multiplier by re-evaluating the
design map about 50 times per iteration, which made the morphology dominate the run time.

The multiscale element needs no derivative of its condensed basis: the Schur complement
$K_c = \Phi^T K_f \Phi$ with $\Phi = [I; -K_{ii}^{-1} K_{ib}]$ satisfies
$dK_c = \Phi^T dK_f\, \Phi$, so the sensitivity is the voxel energy of the reconstructed
fine field. Element-interior Lagrange (bubble) nodes are dropped, since the condensed
interior already contains them.

Open ends: 3D and the GPU matrix-free subvoxel solver of `18_mlhp`, where the dof reduction
matters far more than in these small 2D runs, where the filter time already matches the
coarse FEM time; the morphology is four FFT convolutions per iteration there too.
