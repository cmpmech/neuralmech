# Physics-Informed Neural Networks

Physics-informed neural networks and their variations for **Chapter 13 (Physics-Informed
Neural Networks)**. The physics is written directly in each driver as a residual,
energy density, or weak-form integrand using `DL.differentiate`; `helper.py` only
supplies sampling, the cost of each method, and the training loop, in any dimension.

## Data generation

- `elasticity2D_reference.py` -> `data/elasticity2D_reference.npz`
  conforming mlhp references for perforated plates with 1 to 8 holes per side, and a
  uniform finite element study
- `elasticity2D_nonlinear_reference.py` -> `data/elasticity2D_nonlinear_reference.npz`
  neo-Hookean plate under increasing traction, reference, linear, and coarse solutions
- `poisson_corner_reference.py` -> `data/poisson_corner_reference.npz`
  mlhp multi-level hp refinement study of the corner singularity in 1 to 6 dimensions

## Drivers

- `bar_forward.py`
  forward bar with a manufactured solution, solved as `METHOD` = pinn, dem, vpinn or wan;
  `PROBLEM` = smooth or singular (weak singularity at $x=0$), and `VALIDATION` tracks the
  deep energy method's energy on a ten times finer grid
- `bar_extensions.py`
  forward bar physics-informed neural network with `OPTIMIZER` (adam, lbfgs, elm),
  `SAMPLING` (uniform, random, sobol, adaptive), and self-adaptive weights
- `bar_inverse.py`
  identifies the stiffness $EA$ of the bar from full-field or sparse displacement data; for
  sparse data, `SOLVER` = pinn (second network for $u$) or fem (differentiable finite elements)
- `elasticity2D.py` _needs `elasticity2D_reference.npz`_
  plane stress plate with a hole, pinn or dem, next to the reference and the coarse finite
  element solution with the same energy error
- `elasticity2D_geometry.py` _needs `elasticity2D_reference.npz`_
  deep energy method error over the number of holes and collocation points
- `elasticity2D_nonlinear.py` _needs `elasticity2D_nonlinear_reference.npz`_
  deep energy method for the neo-Hookean plate over the degree of nonlinearity
- `elasticity2D_data.py` _needs `elasticity2D_reference.npz`_
  displacement measurements added to the forward problem, for the deep energy method and
  as a penalty for finite elements
- `poisson_corner.py` _needs `poisson_corner_reference.npz`_
  deep energy method for the Poisson corner singularity of Kopp et al. (2022) in 1 to 6
  dimensions, error and runtime compared against mlhp

## Non-obvious technicalities (authored by Claude)

**Manufactured load.** Only $u$ and $EA$ are specified; the load
$p=-\frac{d}{dx}(EA\,u')$, the Dirichlet value $g=u(0)$, and the Neumann force
$f=EA\,u'(1)$ are obtained with automatic differentiation, so any smooth pair can be
swapped in at the top of the bar drivers.

**Costs.** All methods share one ansatz and training loop and differ only in the cost:
the physics-informed neural network averages squared pointwise residuals, the deep
energy method sums the midpoint-integrated potential energy
$\int_\Omega \frac{1}{2}EA\,u'^2 - p\,u\,dx - f\,u(1)$, and the variational
physics-informed neural network averages the squared weak residuals
$r_k=-\int_\Omega EA\,u' v_k'\,dx+\int_\Omega p\,v_k\,dx+f\,v_k(1)$. Except for the
plain physics-informed neural network, the Dirichlet condition is enforced exactly
through $\hat{u}=g+x\,\mathcal{N}(x)$, as the energy and weak forms assume admissible
trial functions.

**Test functions.** The test functions $v_k=\sin((k-\tfrac{1}{2})\pi x)$ vanish at the
Dirichlet end $x=0$ but not at the Neumann end $x=1$. With $\sin(k\pi x)$, the boundary
term $f\,v_k(1)$ vanishes, and the Neumann condition would never enter the cost.

**Weak adversarial network.** The test function is a second network
$v=x\,\mathcal{N}_v(x)$ that maximizes the cost while $\hat{u}$ minimizes it; the
maximization flips the sign of the test-network gradients before a shared Adam step.
The weak residual is divided by $\|v\|_{H^1}^2=\int_\Omega v^2+v'^2\,dx$. Without a
normalization, scaling up $v$ increases the cost without bound; with only the $L^2$ norm,
an oscillating $v$ keeps $\|v\|_{L^2}$ fixed while $v'$ and thus the residual grow. With
the $H^1$ norm, the maximum over $v$ is the $H^{-1}$ norm of the residual, which vanishes
only at the solution. Simultaneous descent-ascent still cycles; it settles with a ten
times larger learning rate for the test network and an exponential learning rate decay
(`DECAY = 0.999`), which in turn slows the other methods, so they keep a constant rate.

**Sampling.** With the weights $|\Omega|/N$, random and Sobol points turn the cost
into a (quasi) Monte Carlo estimate of the integrated squared residual. Random sampling
draws new points every epoch, so the network cannot overfit a fixed point set but the
cost becomes noisy. Adaptive sampling evaluates the residual on random candidates every
`REFINE_EVERY` epochs and adds the `REFINE_POINTS` worst ones. On the smooth bar,
uniform midpoints remain the most accurate; the alternatives pay off for localized
features or in higher dimensions, where grids become too expensive.

**Self-adaptive weights.** Each collocation and boundary point gets a weight
$\lambda_i$ in $\sum_i\lambda_i r_i^2$, which is maximized while the network minimizes
the cost; it reuses the gradient ascent of the weak adversarial network. Since
$\partial\mathcal{C}/\partial\lambda_i=r_i^2\geq0$, the weights only grow, and they grow
fastest where the residual stays large, which steers the network toward stubborn points.

**Extreme learning machine.** With random, frozen hidden layers, the network output
$\hat{u}=\boldsymbol{\phi}(x)^\intercal\boldsymbol{\beta}$ is linear in the output layer
$\boldsymbol{\beta}$, and for a linear differential equation so are all residuals,
$\mathbf{r}=\mathbf{A}\boldsymbol{\beta}-\mathbf{b}$. A single Gauss-Newton step,
i.e. the least-squares solution, is then exact. `helper.least_squares` assembles
$\mathbf{A}$ row by row with automatic differentiation from the same residual functions
as the physics-informed neural network and solves in double precision, since
$\mathbf{A}$ is ill-conditioned. The hidden weights and biases are drawn from
$[-10,10]$ instead of the usual initialization, which on $[0,1]$ would leave the tanh
features nearly linear and the basis poor.

**Inverse problem.** The bar equation only determines the axial force $EA\,u'$ up to a
constant, which the Neumann residual $EA\,u'(1)-f$ fixes. Where $u'=0$ (here at
$x=0,\tfrac{1}{2},1$), the stiffness drops out of the equation and is only interpolated
by the network. With full-field data this is harmless, but with sparse data the ends
$x=0$ and $x=1$ are where the identified $EA$ deviates. With sparse data, the residual
(of order $p^2\sim10^2$ initially) outweighs the sensor misfit, and the optimizer first
satisfies the physics with a large $EA$ and a flat $u$; `DATA_WEIGHT` lifts the misfit to
a comparable scale. A softplus keeps $EA$ positive.

**Differentiable finite elements.** With `SOLVER = "fem"`, $u$ comes from linear elements
(one per collocation cell) with the network stiffness at the element midpoints, a dense
`torch.linalg.solve`, and linear interpolation at the sensors. Autograd differentiates
through the solve, which is the discrete adjoint method; the governing equation holds
exactly on the mesh, so the cost is the sensor misfit alone and needs no weighting.

**Weak singularity.** `PROBLEM = "singular"` uses $u=x^{0.65}-0.65x$, $EA=1$, the 1D
variant of the corner benchmark below. The energy integrands $u'^2$ and $p\,u$ scale as
$x^{-0.7}$ and are integrable, whereas the squared strong residual scales with
$p^2\sim x^{-2.7}$. Neither method resolves the strain below the first midpoint; the
physics-informed neural network ends up offset over the whole bar (relative $L^2$ error
about 3 times that of the deep energy method on a uniform grid). The test points are
logarithmically spaced to resolve the singularity in the plots.

**Overfitting the quadrature.** With `SAMPLES = 20` and `VALIDATION = True`, the deep energy
method first approaches the exact energy $-3\pi^2/4$ and after about 3000 epochs turns
into a staircase that is flat at the midpoints and steep between them: the training
energy drops without bound, while the validation energy on 200 midpoints rises.

**Plate with a hole.** The perforated plate consists of $k\times k$ cells with a centered
square hole of a fifth of the cell size. The holes lie on cell boundaries of both the
collocation grid and the mlhp mesh (resolutions are multiples of $5k$), so the reference is
boundary-conforming without cut cells and the midpoint quadrature of the deep energy
method integrates the exact domain. The reference refines towards the reentrant hole
corners and the clamped corners; `refineTowardsBoundary` needs a sphere of a tiny positive
radius there, as a radius of 0 does not refine at points inside the plate, and without the
refinement the compliance only converges like the uniform meshes. The deep energy method matches the reference to a
few percent, whereas the physics-informed neural network converges to a wrong field
(relative error near 1) with a small cost. The reentrant hole corners carry stress
singularities $\sigma\sim r^{-0.46}$, so the strong residual $\nabla\cdot\sigma\sim
r^{-1.46}$ of the true solution is large at the nearby collocation points: evaluated at
the deep energy solution, the physics-informed cost is about 11, and training from there
drifts to a smooth wrong field with a cost of about 0.2. Without the hole, both methods
agree. The energy form only requires first derivatives, which remain square-integrable.

**Corner singularity in arbitrary dimensions.** The benchmark of Kopp et al. (2022,
https://doi.org/10.1016/j.cma.2022.115575) solves $-\Delta u=f$ on $[0,1]^d$ with
$u=r^{1/2}$ and $f=\frac{3-2d}{4}r^{-3/2}$, zero flux on the faces $x_i=0$, and Dirichlet
values on the faces $x_i=1$. In one dimension $\sqrt{x}$ has infinite energy, so both
scripts use mlhp's variant $u=x^{0.65}-0.65x$ with $u(0)=0$ and zero flux at $x=1$; its
source $x^{-1.35}$ is not integrable, which is why the one-dimensional network enforces
$u(0)=0$ exactly through $\hat{u}=x\,\mathcal{N}(x)$ instead of a penalty. The deep
energy method suits the problem: the zero-flux faces are natural boundary conditions,
and the energy only involves $\nabla u\sim r^{-1/2}$, whereas the strong residual of a
physics-informed neural network would weigh the source $r^{-3/2}$ at the points closest to
the corner. The mlhp reference refines towards the corner with a uniform degree
$p=\text{depth}+1$, as in the paper, and its error is integrated with Gauss quadrature
on the refined mesh. The network error
$\|\nabla(u-\hat{u})\|_{L^2}/\|\nabla u\|_{L^2}$ is integrated the same way
(`helper.graded_quadrature`): Gauss points on cells halved towards the corner over
`ERROR_LEVELS` levels. Random test points rarely land near the corner, where
$\nabla u\sim r^{-1/2}$ and the network error concentrate; this matters most in one
dimension, where the finest cell of size $h$ still holds a fraction $h^{0.3}$ of the
energy, and little from two dimensions on, where it holds $h^{d-1}$. The timings compare
GPU training (evaluation excluded) with mlhp's assembly and diagonally preconditioned
conjugate gradient solve on all CPU threads (error integration excluded). The default
mlhp build instantiates only up to three dimensions; any higher dimension works after
rebuilding with `-DMLHP_DIMENSIONS=6` (or larger), here also with
`-DMLHP_DEBUG_CHECKS=OFF`. In one dimension the mlhp error only drops by about
$2^{-0.15}$ per level, because the energy of the finest element scales like $h^{0.3}$
for $u'\sim x^{-0.35}$.

**Open points on the corner benchmark.** The mlhp reference reproduces the paper's uniform
tensor curves (e.g. 0.07 % in 4D at depth 3, 0.2 % in 5D at depth 2) but not its second,
graded trunk space strategy (`makeHpTrunkSpace` with `LinearGrading(1)`), which reaches the
same errors with far fewer unknowns and would let 6D refine deeper. In high dimensions the
corner holds almost no energy (in 6D, $10^{-6}$ of it within $r<0.1$), so the global energy
error mostly measures the smooth part; a local error near the corner, and training the
network on `graded_quadrature` instead of Sobol points, would test the singularity itself.

**Energy error.** For any displacement that satisfies the Dirichlet conditions, the
potential exceeds its minimum by half the squared energy norm of the error,
$\Pi(u)-\Pi(u^\ast)=\frac{1}{2}\|u-u^\ast\|_E^2$, and at the minimum
$\Pi(u^\ast)=-\frac{1}{2}c$ with the compliance $c=\mathbf{f}^\intercal\mathbf{u}$
(`helper.energy_error`). The energy error of the network thus only needs its potential,
integrated on a grid finer than the training points, since the network may fit its own
quadrature. For Galerkin finite elements, it reduces to $\sqrt{(c^\ast-c^h)/c^\ast}$. With
data, neither solution minimizes $\Pi$ anymore, but both remain admissible, so the same
error measure applies. The penalty adds
$\frac{\lambda}{2}\sum_m|u(\mathbf{x}_m)-\tilde{u}_m|^2$ to both, i.e.
$(\mathbf{K}+\lambda\mathbf{N}^\intercal\mathbf{N})\mathbf{u}=\mathbf{f}+\lambda
\mathbf{N}^\intercal\tilde{\mathbf{u}}$ for the finite elements, where the rows of
$\mathbf{N}$ hold the shape functions at the sensors, evaluated one unit vector at a time.

**Degree of nonlinearity.** The neo-Hookean energy
$W=\frac{\mu}{2}(\operatorname{tr}\mathbf{C}-2-2\ln J)+\frac{\lambda}{2}(\ln J)^2$ is mlhp's
two-dimensional form, which linearizes to plane strain. The traction sets how nonlinear
the response is, measured by $\|u-u_{\text{lin}}\|_{L^2}/\|u\|_{L^2}$ against the plane strain
solution under the same load. This normalized measure makes the x-axis comparable across
problems. Newton with five load steps fails on the refined reference beyond a traction of
0.4.

**Scores at equal cost.** Error and time are two axes of a work-precision diagram, and a
single ratio such as error per second mixes quantities that do not trade off linearly. The
dimension study instead compares at equal time: the best finite element error reached
within the training time of the network, divided by the network error.
