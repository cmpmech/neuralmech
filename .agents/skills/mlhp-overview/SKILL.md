---
name: mlhp-overview
description: >-
  Essential orientation for the mlhp finite element kernel - read BEFORE writing
  or editing any mlhp code, C++ or Python. Covers what mlhp is, the concepts
  shared by its C++ and Python APIs, the standard workflow, where to find
  authoritative API usage, how to verify results, and which detailed skill to
  use next. Always use first when starting an mlhp task.
---

# mlhp - project overview

This is the orientation map for mlhp: it does not duplicate the focused skills
(immersed methods, visualization, C++ style and utilities) - it points to them.

## What mlhp is

mlhp is an efficient, general-purpose finite element kernel: a C++ core with
extensive Python bindings (on PyPI as `mlhp`, `pip install mlhp`).

Particular strengths:
- **Non-conforming / immersed finite elements** (the Finite Cell Method):
  simulate directly on implicit geometries, STL surfaces, and voxel data with no
  conforming mesh - mlhp's standout capability (skill **mlhp-immersed-methods**).
- **Multi-level _hp_ on hierarchically refined grids** - local refinement,
  high-order bases, automatic hanging-node handling.
- **Self-contained, interoperable Python package** - the wheels need no extra
  dependencies, interoperate with **numpy / scipy / cupy**, and expose C
  interfaces for compiled integrands and materials (cffi / numba), in the spirit
  of Ansys/Abaqus user subroutines.

It also supports **unstructured meshes** and **B-spline patches** (the latter
without local refinement).

## Where to find answers

No formal docs; in order of authority:

1. **Examples** - `examples/*.cpp` / `*.py`, runnable canonical usage. In
   Python, `mlhp.examples()` lists them and `mlhp.example(name)` prints one;
   they also ship in the wheels.
2. **Tests** - `tests/core/` and `tests/system/` cover far more API combinations.
3. **`help(mlhp.fn)`** - all overloads with types (the shared docstring follows
   the signatures). Browse with `[n for n in dir(mlhp) if not n.startswith("_")]`;
   `mlhp.config()` prints build info.
4. **Headers** (`include/mlhp/core/*.hpp`) - their `//!` comments are authoritative.

When something fails or surprises you, read the relevant example or test before
retrying - wrong names, silent failures, and bad topology flags are visible there.

The Python examples ship in the wheel (`mlhp.examples()` / `mlhp.example(name)`),
but the C++ examples, the tests, and the headers live only in the source
repository. With a wheel-only install and no source checkout, clone or browse
the repo to read them - inspect only, do not modify or compile it (see
"Modifying mlhp vs. using it"): https://gitlab.com/hpfem/code/mlhp
(PyPI: https://pypi.org/project/mlhp/).

## Core concepts (shared by C++ and Python)

- **Mesh/cell vs. basis/element.** A mesh and its cells carry only topology and
  geometry (the reference->physical mapping), no FE logic. Finite elements enter
  through a *basis* on a mesh, whose *elements* are the mesh cells (element index
  usually equals cell index). Cells and elements are not objects - only indices
  into a mesh or basis.
- **Per-dimension instantiation.** Types are specialized per spatial dimension:
  C++ templates on `D`, and in Python a separate instantiation per dimension
  with automatic dispatch on the objects you pass. The default builds and PyPI
  wheels instantiate dimensions 1-3 only, so dimensions > 3 work but need a
  custom build.
- **`make*` factories.** Objects are usually built through factory functions
  (`makeRefinedGrid`, `makeHpTrunkSpace` / `makeHpBasis`, ...), not direct
  construction.
- **Fields (spatial functions).** Physics inputs - sources, BC values, material
  coefficients, implicit domains - are scalar/vector/implicit functions. Prefer
  an existing primitive where one fits; otherwise build one from a C++ lambda or
  a Python expression string (`scalarField` / `vectorField` / `implicitFunction`,
  e.g. `"x**2 + y**2 < 9"`).
- **Data layout.** Row-major (C-style) throughout; sparse matrices are CSR.
- **Element-local dof ordering.** Dofs are grouped by field: all of field 0
  (e.g. x-displacement), then field 1, and so on. Within a single field, the
  dofs of a tensor-product space are a C-style linearized tensor product.
- **n-cube face numbering.** `face = axis * 2 + side`: 2D 0=x-min, 1=x-max,
  2=y-min, 3=y-max; 3D adds 4=z-min, 5=z-max. Relevant whenever you touch
  mesh boundaries or element faces.

## The canonical workflow

Most problems follow one arc; transient or nonlinear ones wrap it in a
time/Newton loop:

1. **Mesh** - Cartesian grid (optionally refined), unstructured mesh, or
   B-spline patch; for immersed problems, a background grid filtered against an
   implicit domain.
2. **Basis** - a multi-level _hp_ (or B-spline / unstructured) basis on the
   mesh, given a degree and a field-component count.
3. **Dirichlet dofs** - integrate strong BCs on mesh faces into an
   `(indices, values)` pair.
4. **Assemble** - allocate a sparse matrix and RHS vector, build an *integrand*
   (the physics), integrate over the domain and/or boundary.
5. **Solve** - built-in CG (symmetric) / BiCGStab, or a dense direct solve for
   small systems; then re-insert the Dirichlet dofs.
6. **Postprocess** - VTU output (cell mesh + processors + target), direct
   field/integral probes, or in-memory matplotlib plots (2D Python).

**The two APIs mirror this workflow but sometimes name things differently** -
don't guess one from the other; check the examples, `help()`, or the bindings:

| Step | C++ | Python |
|---|---|---|
| Refined grid | `makeRefinedGrid` | `makeRefinedGrid` |
| _hp_ basis | `makeHpBasis<TrunkSpace>` | `makeHpTrunkSpace` |
| Dirichlet dofs | `boundary::boundaryDofs` | `integrateDirichletDofs` |
| Allocate matrix | `allocateMatrix<...>` | `allocateSparseMatrix` |
| Integrate | `integrateOnDomain` | `integrateOnDomain` |
| Re-insert Dirichlet | `boundary::inflate` | `inflateDofs` |
| VTU output | `writeOutput` | `basisOutput` |

Read the example closest to your problem before writing setup code - it shows
call patterns and parameter choices `help()` does not.

## Verify what you produce

A simulation that runs is not a simulation that is correct - confirm the result
is sensible before reporting it done, rather than trusting that it executed.
Match the check to the task; the cheap ones cost almost nothing:

- **Sanity probes (routine).** Print the basis - `print(basis)` in Python,
  `print( *basis, std::cout )` in C++ (basis held in a smart pointer) - to see
  the element, dof, and field counts; likewise the mesh and the matrix shape.
  Cheap, and informative enough that it can stay in.
- **Pointwise evaluation.** Where you know what to expect, probe the solution
  with `scalarEvaluator` / `vectorEvaluator` / `mechanicalEvaluator`
  (displacement, stress, strain) and check that magnitudes, signs, and boundary
  values are physically sensible. Cheap integral checks help where they apply
  (e.g. total reaction vs. applied load).
- **Visual check (when it fits).** When a picture is the natural check -
  geometry, cut cells, a field distribution, a deformation - or when a figure is
  wanted, render and look; see **mlhp-visualize** (2D Python can plot in memory).
  Not every run needs this.

Remove checks that were only scaffolding once they have served their purpose;
keep output that stays genuinely informative on its own (a basis or mesh
summary, say).

## Other mlhp skills

Their own descriptions say when to invoke them; in short:
- **mlhp-immersed-methods** - immersed / finite-cell methods.
- **mlhp-visualize** - rendering and verifying results visually.
- **mlhp-cpp-style** - writing or editing mlhp C++.
- **mlhp-cpp-utilities** - existing C++ helpers before reimplementing one.

## Modifying mlhp vs. using it

- **Using mlhp** (most C++ projects embed it; Python users via the wheel): do
  not modify mlhp itself unless explicitly asked - treat it as a fixed
  dependency with a stable API. Your own application code carries no such
  constraint.
- **Developing mlhp**: the public API (core headers, the Python bindings, and
  `mlhp.py`) has downstream consumers, so change it only when clearly necessary,
  and flag breaking changes deliberately.

## Common traps

- **Never modify a mesh after building a basis on it** - the basis caches
  topology and dof maps; refine first, then build the basis.
- **Dirichlet-reduced systems** - if Dirichlet dofs were excluded at matrix
  allocation, the solution vector lacks them until you inflate it back.
- **Moment-fitting quadrature on non-polynomial integrands** - it assumes a
  polynomial and silently produces wrong integrals; use space-tree quadrature
  (see mlhp-immersed-methods).
- **Full vs. leaf indices** on refined grids - the *full* index counts parents,
  the *leaf* index only active leaves; mixing them silently indexes the wrong
  cells.