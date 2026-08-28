# Solvers

Classical physics solvers the drivers in [`projects/`](../projects/README.md) build on.

## Submodules

- [`cuwave/`](cuwave) -- GPU finite difference wave solver with differentiable adjoints,
  behind the wave and transient acoustic topology optimization drivers. Install with
  `pip install -e solvers/cuwave`, then import as `from cuwave.wave import ...`.
- `mlhp_source/mlhp` -- hp finite element kernel. `pip install mlhp` covers the Python
  API; advanced physics needs the C++ build.

## Modules

| file | description |
| --- | --- |
| `optimization.py` | MMA, structured-grid FEM assemble-and-solve, conic density filter, Heaviside projection |
| `homogenization.py` | KUBC energy homogenization of a two-phase cell |
| `truss.py` | linear-elastic truss with a global stiffness matrix |
| `dynamic_mdof.py` | multi-degree-of-freedom spring-mass-damper chain |
| `multibody.py` | planar multibody trees integrated through their Lagrangian equations |
| `bouncing_balls.py` | impulse-based ball contact under gravity |
| `mklwrapper.py` | MKL pardiso factorization bindings |
