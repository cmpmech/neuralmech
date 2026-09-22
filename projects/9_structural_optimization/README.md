# Structural optimization

Optimization and regularization for **Chapter 9 (Numerical Methods)**, built around the
compliance problem: how a constrained optimum is reached, and what has to be added to
the relaxed problem before its solution is meaningful.

## Drivers

`constrained_optimization.py`
    the penalty method, the augmented Lagrangian, the optimality criteria update and MMA
    on a separable test problem with a closed-form optimum

`penalization.py`
    Tikhonov and total variation evaluated on a plate with a hole, once with a wide and
    once with a narrow transition across the boundary

`fwi_penalization.py`
    full waveform inversion of a plate with a central hole from a transducer array on
    the top edge, run once without and once with a total variation penalty, exporting
    the truth and the two reconstructions

`reparametrization.py`
    four disks of ascending radius mapped through the density filter, the smoothed
    Heaviside projection and the SIMP interpolation, exporting one square image per
    stage

`mesh_dependence.py`
    the half MBB beam on a ladder of design resolutions, filtered at the element scale
    only, exporting one image per resolution

`robustness_reference.py`
    the ceiling design of Chapter 8 optimized for the intermediate threshold alone, then
    evaluated as optimized and uniformly dilated, exporting the sound pressure level of
    both

`robustness_robust.py`
    the same design under the robust formulation over an eroded, an intermediate and a
    dilated threshold, with the same dilation and the same export

`erosion_dilation.py`
    a compliance design of a 5:1 cantilever, filtered once and thresholded at the eroded,
    the intermediate and the dilated threshold, exporting one image each

## Non-obvious technicalities (authored by Claude)

### `reparametrization.py`

The radii are picked from a closed form rather than by trial. Filtering a solid disk of
radius $a$ with the conic kernel $w(r)=r_f-r$ leaves the center value

$$\tilde{\gamma}(0)=\frac{\int_0^a (r_f-r)\,2\pi r\,\dd r}{\int_0^{r_f}(r_f-r)\,2\pi r\,\dd r}
=3t^2-2t^3,\qquad t=\frac{a}{r_f},$$

the smoothstep function. It crosses the projection threshold $\eta=1/2$ at exactly
$t=1/2$, so a disk smaller than half the filter radius cannot survive the projection and
a larger one can. `RADII` sits at $t=0.33$, $0.60$, $0.87$ and $1.20$, which places one
disk below the threshold and three above it, and the driver prints the values so the
prediction can be checked ($0.261$, $0.644$, $0.951$, $1.000$ against $0.259$, $0.648$,
$0.951$, $1.000$ from the formula).

`BETA` is a compromise rather than a maximum. A sharper projection would drive the three
surviving disks to exactly one and leave nothing for the interpolation to do, so the
fourth image would repeat the third. At $\beta=8$ the second disk projects to $0.91$ and
the power law takes it to $0.75$, which is the only visible difference between the two
images. In an actual optimization $\beta$ would be raised by continuation instead of
held fixed.

`RESOLUTION` is bounded from below by the layout: with centers on the quarter points,
the largest disk plus the filter radius has to fit inside a quarter of the image, so
$N>4(r_{\max}+r_f)$. Everything scales with $r_f$, so changing `FILTER_RADIUS` alone
changes only how much of the frame the design occupies, not the result.

### `fwi_penalization.py`

The driver is the cuwave example `examples/fwi/regularization/scalar2D_fwi_adam_penalty.py`
reduced to the one comparison the chapter needs, with L-BFGS in place of Adam because the
line search returns a cleaner reconstruction in far fewer iterations.

The measurement is simulated on its own grid and its own stencil -- $383^2$ nodes at
space order $8$ against $256^2$ at order $4$ for the inversion -- so the record the
inversion fits was never produced by the operator it inverts. Nothing but the physical
coordinates is shared: `point_source` divides the signal by the cell volume, so the
source amplitude is independent of the spacing, and the record is resampled from
$\Delta t_{\textrm{obs}}$ onto the inversion time grid. The odd $383$ is deliberate,
so the two grids share no nodes.

The array sits entirely on the top edge, which is what makes the figure worth showing:
the inversion sees the hole only in reflection. At `T = 2.5` traversals only the
illuminated top arc of the hole returns, and the two runs are indistinguishable; at
`T = 4.0` the record is long enough for the wave to travel around the hole and the full
outline closes. The interior of the hole never fills in -- a single-sided array carries
no information about what is behind the first interface -- so the reconstruction is an
outline, not a disk.

`WEIGHT` is set against `ITERS`, not independently. The penalty acts on every iteration,
so the same weight that cleans up a $20$-iteration run over-smooths a longer one. At
$3\times10^{-3}$ and $40$ iterations the speckle between the transducers and the hole is
gone while the outline survives; at $10^{-2}$ the outline goes with it.

The misfit is the honest part of the comparison, and it runs against the figure. The
unpenalized inversion reaches $0.179$ of its starting misfit, the penalized one only
$0.300$ -- the run that fits the data better is the one that looks worse. That is the
whole argument for the penalty: past a point the remaining misfit is artifacts and
discretization error, and an optimizer with nothing holding it back spends its freedom
fitting them into the model. The segmentation scores do not separate the two either
($f_1$ of $0.104$ against $0.108$), because the penalty pulls the outline toward the
threshold as fast as it removes false positives.

The penalty has to enter the objective as well as the gradient. L-BFGS backtracks on the
value, so a line search that measured only the misfit would reject the steps the penalty
gradient asks for.

### `mesh_dependence.py`

The driver is the compliance loop of `projects/8_physics_drivers/topopt_elasticity2D.py`
run once per entry of `RESOLUTIONS`, with the filter radius quoted in elements instead of
in length. Every other setting is held fixed, so the design grid is the only thing that
changes between the panels.

`RMIN = 1.5` is the whole point of the driver and is not a regularization. A radius fixed
in length imposes a minimum feature size and makes the design mesh convergent; a radius
of one and a half elements shrinks with the mesh, so it carries no length of its own and
the design stays free to become finer at every refinement. What it does remove is the
checkerboard, which is an element-scale artifact of the $Q_1$ element overestimating the
stiffness of an alternating density pattern. The filter also smooths the design space and
keeps the optimizer out of poor local minima, so it earns its place even here
(https://doi.org/10.1007/s001580050176).

`RMIN` has a floor. The conic weights are $w=r_f-\|\mathbf{x}_i-\mathbf{x}_j\|$ in
element units, so at `RMIN = 1.0` the four neighbors get weight zero and the filter is
the identity. `1.2` is the smallest value that filters at all; `1.5` spans the full
$3\times3$ stencil, with the corners at $1.41$ carrying a small weight.

`DEGREE = 1` rather than the cubic elements of the Chapter 8 driver. Cubic elements also
remove the checkerboard, but they remove the mesh dependence with it: the artificial
stiffness is what rewards fine structure, so at `DEGREE = 3` the four panels converge to
one design with better resolved boundaries and the figure loses its argument. Bilinear
elements plus an element-scale filter separate the two effects, and they run the whole
ladder in about two minutes against half an hour.

The member count grows roughly $2 \to 6 \to 9 \to 14$ across the four grids while the
thresholded compliance stays near $185$--$192$. That is the honest statement of the
problem: the designs differ, their performance does not, so nothing in the objective
selects one of them.

The three finer runs stop on `MAX_ITER` rather than on `CHANGE_TOL`, since individual
elements keep flipping along the member edges long after the layout has settled.

### `robustness_reference.py` and `robustness_robust.py`

Both drivers are the Adam loop of `projects/8_physics_drivers/topopt_helmholtz2D.py` with
every setting shared, so the formulation is the only thing that changes between them.
The perturbation is a uniform dilation: the final filtered field is projected at
`ETA_PERTURBED = 0.4` instead of $0.5$, which moves every boundary outward by the same
distance, the over-etching the robust formulation models.

The room is $18\times9$ instead of the square of the Chapter 8 driver, so the frequency
is moved from the $(5,5)$ mode of the square to the $(6,2)$ mode of the rectangle,
$f=\frac{c}{2}\sqrt{(n/L_x)^2+(m/L_y)^2}=68.77$. Keeping $67.43$ lies between two modes
of the rectangle; it separates the two formulations even more ($67.5\to103.4$\,dB
against $74.3\to74.3$\,dB), but the frequency then no longer means anything.

`RMIN = 6` rather than the $2$ of the Chapter 8 driver, in both files. A threshold shift
moves an edge by a fraction of the filter radius, so at $r_f=2$ the eroded and dilated
designs differ from the intermediate one by less than a voxel and the robust formulation
has nothing to act on.

The robust thresholds are $0.6$, $0.5$ and $0.4$, so the perturbation is one of the
sampled designs. The guarantee holds only at the samples: on the square room, $0.7$,
$0.5$ and $0.3$ returned a design at $77.4$\,dB as optimized and $84.1$\,dB at
$\eta=0.4$, since nothing controls the cost between the thresholds. With $0.4$ sampled
the robust design reaches $78.2$\,dB as optimized and $77.7$\,dB dilated, against the
$73.2$ and $95.7$\,dB of the reference one: the robust design gives up $5$\,dB at the
intermediate threshold and gains $18$\,dB under the perturbation.

The maximum over the three thresholds is not smoothed or bounded. Each iteration steps
along the gradient of whichever design is currently worst, and Adam's moment estimates
average over the switches. This takes three state solves per iteration and roughly
twice the wall-clock time of the reference run.

### `erosion_dilation.py`

The design is optimized with the density filter in the loop rather than the sensitivity
filter of `mesh_dependence.py`, so the filtered field the three thresholds act on is the
one the optimizer actually saw. The thresholds are then applied as hard cuts instead of
through the smoothed Heaviside projection, since a steep projection leaves grey ghosts
where a member is about to vanish and the figure is meant to be black and white.

`ETA_E, ETA_D = 0.6, 0.4` shifts every boundary without breaking a member: the eroded
compliance rises by $9\,\%$ ($786\to857$) and the dilated one drops by $6\,\%$ ($741$).
At $0.65$ and $0.35$ the thinnest members already break ($6716$), and at $0.75$ and
$0.25$ nearly every member is cut ($3.1\times10^4$), which reads as a broken design
rather than a perturbed one. The compliance is evaluated on the element grid;
`UPSAMPLE` only smooths the exported boundary.

A uniaxial tension member was tried first and discarded: pulled at the middle of the
right edge, the optimum is a solid bar forking into a Y, with no thin members or gaps
for the thresholds to act on.
