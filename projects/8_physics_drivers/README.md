# Governing equations

Small reference solvers for the governing equations of mechanics, supporting
**Chapter 8 (Governing Equations)**. Each driver pairs a standalone solver in
`solvers/` with a minimal example that sets up a problem, solves it, and plots
the result.

## Drivers

`balls_example.py`
    Balls bouncing under gravity with wall and pairwise contact (impulse-based).

`contact_blocks.py`
    Elastic square blocks falling under gravity and colliding, simulated with a
    Warp MLS-MPM solver and drawn as deformed surfaces shaded by von Mises stress.

`truss_example.py`
    Linear-elastic arch truss solved with a global stiffness matrix.

`mdof_example.py`
    Multi-degree-of-freedom spring-mass-damper chain under harmonic forcing.

`triple_pendulum_example.py`
    Chaotic triple pendulum built from the generic planar multibody solver
    (an open kinematic tree integrated through its Lagrangian equations of motion).

`cartpole_example.py`
    Inverted pendulum on a cart from the same multibody solver (a prismatic cart
    carrying a revolute pole), released just off vertical with no control.

`poisson2D_example.py`
    Poisson problem on a 2D mlhp finite-element mesh.

`advection_diffusion2D_example.py`
    Steady advection-diffusion of a Gaussian source on a 2D mlhp finite-element mesh.

`elastic2D_example.py`
    Linear elasticity on a 2D mlhp finite-element mesh.

`helmholtz2D_example.py`
    Time-harmonic acoustics (Helmholtz) on a 2D mlhp finite-element mesh.

`waveND_example.py`
    Scalar wave equation via the finite-difference CuPy wave solver.

`heatND.py`
    Implicit (backward-Euler) transient heat conduction on an N-D mlhp
    finite-element mesh; an initial hot-spot diffuses to zero.
