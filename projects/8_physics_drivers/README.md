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
