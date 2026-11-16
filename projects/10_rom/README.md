# Reduced Order Models and Classical Surrogates

Reduced order models and classical surrogates for **Chapter 10 (Machine Learning in
Computational Mechanics)**. A cantilever of Hermite beam elements is the shared full
order model of the beam drivers; the heat driver uses 2D finite differences.

## Drivers

- `rom_beam.py`
  tip pulse on a cantilever: Galerkin reduced models from POD and from eigenmodes,
  error over the rank $R$, mode shapes, and timing
- `rom_nonlinear_heat.py`
  transient heat with temperature-dependent conductivity: POD against the eigenmodes of
  the linearized problem, reduced operator precomputed as a tensor
- `surrogates_beam.py`
  first eigenfrequency over the stiffness of the two beam halves: quadratic response
  surface, kriging and polynomial chaos from 20 Latin hypercube samples
- `helper.py`
  beam stiffness and consistent mass matrices, average acceleration Newmark integration

## Non-obvious technicalities (authored by Claude)

### Time integration of the beam

The explicit central difference scheme is stable only for
$\Delta t < 2/\omega_{\textrm{max}}$. With 100 Hermite elements and the consistent mass
matrix, $\omega_{\textrm{max}} \approx 6\cdot10^5$ (for unit stiffness, mass and
length), i.e., $\Delta t < 3\cdot10^{-6}$, which would take $7\cdot10^6$ steps for the
simulated 20 time units. Both the full and the reduced models therefore use the
unconditionally stable average acceleration Newmark scheme ($\beta=1/4$,
$\gamma=1/2$) with $\Delta t = 10^{-3}$, about 30 steps per period of the fifth mode.
The reduced model alone could also run explicitly with a much larger step, as its
highest frequency is that of the last retained mode.

### POD in the mass inner product

The beam dofs mix deflections and rotations. A plain SVD of the snapshot matrix weighs
them by their size, and since the rotations of the higher modes are large, its leading
modes are a blend of eigenmodes. The first such mode has a Rayleigh frequency of 3.60
instead of 3.516, and the reduced model drifts out of phase: a relative error of 88 %
at $R=1$ over 20 time units, against 17 % for the first eigenmode. The driver therefore
computes POD in the mass inner product $\mathbf{u}^\top\mathbf{M}\mathbf{v}$: with
$\mathbf{M}=\mathbf{L}\mathbf{L}^\top$, the SVD of $\mathbf{L}^\top\mathbf{X}$ yields
$\tilde{\mathbf{U}}$, and the basis is $\mathbf{L}^{-\top}\tilde{\mathbf{U}}$. For the
undamped linear beam these modes reproduce the eigenmodes, and both bases give the same
errors up to the sixth mode. Beyond, the pulse barely excites the modes within the
training window, the POD modes are fitted to noise, and the error stalls near
$10^{-3}$, while the eigenmodes keep converging.

### Reduced operator of the nonlinear heat problem

The face conductivity $k=1+\beta\bar{u}$ uses the temperature averaged onto the face,
$\bar{\mathbf{u}}=\mathbf{A}\mathbf{u}$, so the stiffness
$\mathbf{K}(\mathbf{u})=\mathbf{D}^\top\textrm{diag}(1+\beta\mathbf{A}\mathbf{u})\mathbf{D}$
is affine in $\mathbf{u}$. With $\mathbf{u}=\boldsymbol{\Phi}\mathbf{c}$,

$$\boldsymbol{\Phi}^\top\mathbf{K}\boldsymbol{\Phi}
= (\mathbf{D}\boldsymbol{\Phi})^\top\mathbf{D}\boldsymbol{\Phi}
+ \beta\sum_j c_j\,(\mathbf{D}\boldsymbol{\Phi})^\top\textrm{diag}(\mathbf{A}\boldsymbol{\phi}_j)\,\mathbf{D}\boldsymbol{\Phi},$$

where the third-order tensor of the second term is computed once (`einsum`). An online
step then costs $\mathcal{O}(R^3)$ and never touches the full grid. A general
nonlinearity has no such expansion and needs hyper-reduction (e.g., DEIM).

### Why the linearized eigenmodes stall

The eigenmodes of the linear problem are sines that vanish at the boundary together
with their second derivatives. The linear solution shares this property (there,
$\Delta u = -p \approx 0$ at the boundary), and its sine series converges quickly. In
the nonlinear problem, $k\Delta u + \beta\lvert\nabla u\rvert^2 = -p$, so $\Delta u \neq 0$
at the boundary. The series then converges only algebraically, and the error of the
eigenmode basis stalls near 3 % up to $R=30$ (0.6 % even with 200 modes in a test), with
the largest errors along the boundary. POD learns the actual shape and reaches
$4\cdot10^{-5}$ at $R=8$.

### Polynomial chaos statistics

The inputs are mapped to $\xi\in[-1,1]^2$ and expanded in products of Legendre
polynomials $P_i(\xi_1)P_j(\xi_2)$ of total degree $i+j\leq3$. For uniform $\xi$,
$\mathbb{E}[P_iP_k]=\delta_{ik}/(2i+1)$, so the mean is the constant coefficient and the
variance is $\sum c_{ij}^2/((2i+1)(2j+1))$ over all other coefficients, without any
sampling.
