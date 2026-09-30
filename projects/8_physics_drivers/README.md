# Governing equations

Reference solvers for the governing equations of mechanics, supporting
**Chapter 8 (Governing Equations)**, together with classical inverse and
topology-optimization drivers built on top of them.

## Drivers

`balls2D.py`
    Balls bouncing under gravity with wall and pairwise contact (impulse-based).

`truss2D.py`
    Linear-elastic arch truss solved with a global stiffness matrix.

`mdof1D.py`
    Multi-degree-of-freedom spring-mass-damper chain under harmonic forcing.

`triple_pendulum.py`
    Chaotic triple pendulum built from the generic planar multibody solver
    (an open kinematic tree integrated through its Lagrangian equations of motion).

`cartpole.py`
    Inverted pendulum on a cart from the same multibody solver (a prismatic cart
    carrying a revolute pole), released just off vertical with no control.

`poisson2D.py`
    Poisson problem on a 2D mlhp finite-element mesh.

`poisson2D_python.py`
    The same Poisson problem on the pure-Python structured FEM (solvers/FEM),
    with the hole resolved by voxelized finite cells instead of cut-cell quadrature.

`advection_diffusion2D.py`
    Steady advection-diffusion of a Gaussian source on a 2D mlhp finite-element mesh.

`fluid2D.py`
    Incompressible flow past a cylinder in a channel at Re 100 (vortex shedding) on the
    cufluid lattice Boltzmann GPU solver, with a velocity inlet and a pressure outlet.

`elasticity2D.py`
    Linear elasticity on a 2D mlhp finite-element mesh.

`elasticity2D_python.py`
    The same elasticity problem on the pure-Python structured FEM (solvers/FEM):
    voxelized finite cells, face traction, stresses recovered from the gradients.

`homogenization2D.py`
    Effective stiffness of a two-phase inclusion cell by the finite cell method,
    computed with the reusable KUBC energy homogenizer in solvers/homogenization.py.

`plasticity2D.py`
    Small-strain J2 plasticity around a hole in a stretched plate, load-stepped
    with a Newton solve. The return mapping is a C user-material subroutine
    (solvers/material_subroutines/j2.py) compiled at runtime and passed to mlhp
    through its constitutive-equation C-interface.

`plasticity2D_python.py`
    The same J2 plasticity problem on the pure-Python structured FEM (solvers/FEM):
    the return mapping is a vectorized NumPy usermat evaluated over all quadrature
    points at once, with history stored directly at the points.

`fracture2D.py`
    AT2 phase-field fracture of the single-edge notched shear test (Miehe et al. 2010),
    staggered, with the volumetric-deviatoric split. Degraded material, damage equation and
    history update are C subroutines (solvers/material_subroutines/phasefield.py) handed to
    mlhp; bilinear elements refined along the notch and in the lower right quadrant, the
    notch as nearly void elements, pardiso through mlhp.mkl.

`helmholtz2D.py`
    Time-harmonic acoustics (Helmholtz) on a 2D mlhp finite-element mesh.

`waveND.py`
    Scalar and acoustic wave equations in 1D/2D/3D on the cuwave finite-difference
    GPU solver (the `cuwave` package).

`heatND.py`
    Implicit (backward-Euler) transient heat conduction on an N-D mlhp
    finite-element mesh; an initial hot-spot diffuses to zero.

`param_identification.py`
    Inverse identification of the axial stiffness EA(x) of a 1D bar from measured
    displacements, solved as a regularized linear system.

`beam_shape2D.py`
    Laser beam shape optimization for powder bed fusion (Holla et al. 2023): the top-edge
    intensity of a moving laser is shaped by L-BFGS-B with adjoint gradients so that the
    steady advection-diffusion temperature matches a target (a known Gaussian beam or a
    uniform melt pool).

`beam_shape2D_nonlinear.py`
    The same beam shaping with the paper's nonlinear material: temperature-dependent
    conductivity and heat capacity plus latent heat as an apparent heat capacity, solved
    by Newton with a numba-compiled mlhp integrand and the transposed tangent as adjoint.

`topopt_elasticity2D.py`
    Compliance minimization of the half MBB beam with SIMP and an
    optimality-criterion update on a preintegrated structured mlhp grid.

`topopt_poisson2D.py`
    Volume-to-point heat-conduction topology optimization with SIMP, robust
    three-field projection (eroded/intermediate/dilated), beta-continuation, and MMA.

`topopt_mechanism2D.py`
    Compliant force-inverter mechanism synthesis with MMA and penalization
    continuation; port springs and passive patches anchor the input/output nodes.

`topopt_helmholtz2D.py`
    Ceiling acoustic topology optimization minimizing the mean square pressure in a
    quiet box (time-harmonic, own complex FEM assembly, Adam on the adjoint gradient).

`topopt_elasticity2D_cg.py`
    The MBB driver solved with Jacobi-preconditioned conjugate gradients instead of a
    factorization, warm-started across designs (USE_CUPY switches from mlhp CG on the
    CPU to cupyx CG on the GPU). Iteration counts grow with the SIMP contrast, so the
    direct variants stay faster at this problem size.

`topopt_acoustic2D.py`
    Transient acoustic topology optimization: a sine burst travels down a channel and
    the material in two design blocks is shaped to silence (or amplify) a target box,
    with adjoint sensitivities from cuwave's boundary-reconstruction adjoint and MMA
    updates (USE_ADAM switches to Adam).

`topopt_multiphysics2D.py`
    Multiphysics heat sink: minimum thermal compliance (uniform heat source, sink in the
    middle of the left edge) under an upper bound on the structural compliance (left edge
    clamped, point load at the middle of the right edge), with MMA on two constraints.

## Non-obvious technicalities (authored by Claude)

**Beam shaping in 2D.** `beam_shape2D.py` reduces the three-dimensional setup of Holla et
al. (2023) to the longitudinal cross-section (scan direction $x$, depth $y$), so the beam
is a line laser with a one-dimensional profile $u(x)$, and the material is linear
(constant conductivity and heat capacity, no latent heat). In the laser frame the
temperature solves $v \partial_x T - \kappa \Delta T = 0$ with $\kappa \partial_n T =
\alpha u / (\rho c)$ on the top edge, $T = T_0$ at the inflow and zero flux elsewhere.
With $u = \sum_i N_i(x) \beta_i$ on hat functions, the discrete forward problem is linear,
$K T = f_0 + B \beta$. For $J = \tfrac{1}{2} (T - T_d)^\top M (T - T_d)$ the adjoint is
$K^\top \lambda = M (T - T_d)$ and $\partial J / \partial \beta = B^\top \lambda$, the
linear limit of the paper's adjoint equations. $K$ is factorized once, so each L-BFGS-B
iteration costs two back-substitutions. The melt pool target $T_d$ follows the paper's
Case 2: a forward solve without laser in which the temperature inside the box is pinned to
$T_{pool}$ by an $L^2$ penalty. Because this target is not reachable, the loss settles at a
nonzero floor, and the unregularized profile is spiky (the upstream end of the control
window and the melt pool front); the paper smooths the gradient for this reason.

**Nonlinear variant.** `beam_shape2D_nonlinear.py` uses the paper's material law,
$c(T) = \rho c_s(T) + \rho L f_{pc}'(T)$ with $f_{pc} = \tfrac{1}{2}(\tanh((T - T_m)/T_\sigma)
+ 1)$, $T_m = (T_l + T_s)/2$, $T_\sigma = S (T_l - T_s)/2$, and linear $c_s(T)$, $k(T)$. All
heat capacities are divided by $\rho c_s(0)$, so the conductivity carries units of
$\mu m^2/\mu s$. The residual $R(T, \beta) = \int N_i c(T) v \cdot \nabla T + \nabla N_i \cdot
k(T) \nabla T \, d\Omega - B \beta$ is solved by Newton with the consistent tangent
$\partial R / \partial T$, which adds $c'(T) N_j v \cdot \nabla T$ and $k' N_j \nabla T$ to the
linear operator. The integrand reads the current temperature from the dof vector passed as
`data` and the element location map. Instead of deriving the continuous adjoint (the
paper's Eqs. 13 to 15), the driver takes the discrete one: $(\partial R / \partial T)^\top
\lambda = M (T - T_d)$ and $\partial J / \partial \beta = B^\top \lambda$, reusing the LU factors
of the last Newton step. Each optimizer iteration warm-starts Newton from the previous
temperature, so two or three Newton steps suffice.


**Multiphysics heat sink.** `topopt_multiphysics2D.py` shares one design grid between a
scalar (temperature) and a vector (displacement) mlhp basis, each with its own
`StructuredFEM`. Two constraints (volume and structural compliance) exceed the
single-constraint dual subsolver of `solvers.optimization.MMA`, so the driver calls
`mmapy.mmasub` directly. The compliance bound is a fraction of the uniform initial
design's structural compliance: the unconstrained heat sink is a dendritic tree that does
not reach the load, so its structural compliance is that of the void and gives no useful
scale. Both compliances and the volume are normalized to order one before MMA.
