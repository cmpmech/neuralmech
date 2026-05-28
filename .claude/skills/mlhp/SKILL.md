---
name: mlhp
description: General reference for the mlhp finite-element library (C++ core + `pymlhpcore` Python bindings) pinned in `solvers/mlhp`. Covers the full Python API — meshes/grids, refinement, implicit CSG geometry, hp/B-spline bases, scalar/vector fields, quadrature (incl. moment-fitting FCM), elasticity/Poisson/transient integrands, custom compiled integrands, Dirichlet/Neumann BCs, sparse + matrix-free assembly, CG/BiCGStab solvers, postprocessing/VTU/error indicators, triangulation/STL, and the numbering/orientation conventions (face IDs, voxel/cell ordering). Invoke before writing or editing any mlhp driver, including everything in `projects/16_elastic_fem/`.
---

mlhp is a C++ multi-level *hp*-FEM library (arbitrary dimension) with Python bindings exposed as the `pymlhpcore` extension and a thin `mlhp.py` wrapper. Import is always:

```python
import mlhp
```

`pip install mlhp` provides the `pymlhpcore` wheel (the primary import). The `solvers/mlhp` submodule is the pinned C++ source (0.1.2) — read it for exact signatures, but you do not need to build it to run drivers. Advanced physics / custom C++ integrands require the C++ build.

The library is **dimension-templated**: nearly every constructor takes a dimension or infers it from its arguments; objects of different `D` cannot be mixed. Keep `D` (or `ndim`) a single top-level constant in a driver.

---

## Standard workflow

The canonical pipeline (mirrors the C++ examples):

```
Mesh creation ─► Basis definition ─► Dirichlet conditions ─┐
                                  └─► Sparse system alloc ──┤
                Problem physics (integrand) ───────────────┤
                                                            ▼
                                                       Assembly ─► Linear solve ─► Postprocessing
```

A minimal 3D linear-elastic driver (from `examples/elastic_fcm.py`, structured-grid variant):

```python
import mlhp

D = 3
degree = 2
nelements = [10] * D
lengths = [1.0] * D

# 1. Mesh + basis
mesh = mlhp.makeRefinedGrid(nelements, lengths)
basis = mlhp.makeHpTrunkSpace(mesh, degree=degree, nfields=D)

# 2. Dirichlet BCs — clamp face 0 (x-), all components
dirichlet = mlhp.integrateDirichletDofs(mlhp.vectorField(D, [0.0] * D), basis, [0])

# 3. Physics
E  = mlhp.scalarField(D, 200e9)
nu = mlhp.scalarField(D, 0.3)
rhs = mlhp.vectorField(D, [0.0, 0.0, -78.5e3])           # body force
kinematics  = mlhp.smallStrainKinematics(D)
material    = mlhp.isotropicElasticMaterial(E, nu)
integrand   = mlhp.staticDomainIntegrand(kinematics, material, rhs)

# 4. Allocate + 5. assemble
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)

# 6. Solve
P = mlhp.diagonalPreconditioner(matrix)
interior, norms = mlhp.cg(matrix, vector, M=P, maxiter=2000, residualNorms=True)
allDofs = mlhp.inflateDofs(interior, dirichlet)

# 7. Postprocess
processors = [mlhp.solutionProcessor(D, allDofs, "Displacement")]
postmesh = mlhp.gridCellMesh([degree + 2] * D)
mlhp.basisOutput(basis, postmesh, mlhp.PVtuOutput(filename="out"), processors)
```

---

## Class hierarchy

```
AbsMesh
├── AbsGrid
│   └── CartesianGrid                 (makeGrid)
└── AbsHierarchicalGrid               (holds an AbsGrid as base)
    ├── RefinedGrid                   (makeRefinedGrid)
    └── FilteredRefinedGrid           (makeFilteredGrid)

AbsBasis
├── MultilevelHpBasis                 (makeHpTrunkSpace / makeHpTensorSpace; holds an AbsHierarchicalGrid)
├── UnstructuredBasis                 (makeUnstructuredBasis)
├── ElementFilterBasis                (makeFilteredBasis)
└── DummyBasis                        (makeDummyBasis)

AbsSparseMatrix
├── SymmetricSparseMatrix
└── UnsymmetricSparseMatrix
```

A `MultilevelHpBasis` is built on an `AbsHierarchicalGrid` (a refined grid), which wraps an `AbsGrid` base. So the usual chain is `makeGrid → makeRefinedGrid → makeHpTrunkSpace`. `makeRefinedGrid(ncells, lengths)` is a shortcut that builds the Cartesian base internally.

---

## Numbering & orientation conventions

**Spatial axes**: `x = axis 0`, `y = axis 1`, `z = axis 2`. Coordinate arrays/lengths/ncells are always `[x, y, z]` order.

**Box face IDs** — `face(axis, side) = 2*axis + side` (verified in `core/boundary.cpp`). Side 0 is the lower (minus) face, side 1 the upper (plus). Same in 2D and 3D:

| ID | Face | axis | side |
|----|------|------|------|
| 0 | x− (left)   | 0 | 0 |
| 1 | x+ (right)  | 0 | 1 |
| 2 | y− (bottom) | 1 | 0 |
| 3 | y+ (top)    | 1 | 1 |
| 4 | z− (back)   | 2 | 0 |
| 5 | z+ (front)  | 2 | 1 |

So `ifield = faceID // 2` gives the field component normal to that face — used to constrain only the normal displacement (roller BC): `integrateDirichletDofs(..., [face], ifield=face // 2)`.

**Cell / voxel ordering** — for a structured grid with `ncells = [Nx, Ny, Nz]`, cell index runs C-order with the **last axis fastest**: `icell = (ix*Ny + iy)*Nz + iz`. This matches `numpy.ravel("C")` of an array shaped `(Nx, Ny, Nz)`. Verified against mlhp's `findVoxel` in `core/spatial.cpp`: `scalarFieldFromVoxelData(data, nvoxels=ncells, lengths=lengths)` expects `indicator.ravel("C")`, and `basis.locationMaps()` element rows are in the same order.

**Element freedom tables** — `np.array(basis.locationMaps())` has shape `(n_elem, ndof_e)`; row `i` is the global DOF indices for element `i`, in the same element order as above.

**DOF interleaving** — for a vector basis (`nfields=D`) the solution vector is field-interleaved per node: `[u0_x, u0_y, u0_z, u1_x, ...]`. Slice with `sol[0::D]`, `sol[1::D]`, … to separate components. `solutionProcessor` output in a `DataAccumulator` is interleaved the same way.

---

## API reference by category

### Meshes & grids

| Function | Returns | Notes |
|----------|---------|-------|
| `makeGrid(ncells, lengths=1, origin=0)` | `CartesianGrid` | `ncells=[Nx,Ny,Nz]`; `lengths`,`origin` per-axis. |
| `makeGrid(ticks)` | `CartesianGrid` | Non-uniform: `ticks` = per-axis coordinate lists. |
| `makeRefinedGrid(ncells, lengths=1, origin=0)` | `RefinedGrid` | Builds Cartesian base + hierarchical wrapper. |
| `makeRefinedGrid(grid)` | `RefinedGrid` | Wrap an existing grid. |
| `makeRefinedGrid(grid, relativeDepth, maxdepth=∞)` | `RefinedGrid` | Uniform refinement. |
| `makeFilteredGrid(grid, domain=, nseedpoints=4)` | `FilteredGrid` | Keep cells intersecting an implicit domain. |
| `makeFilteredGrid(grid, cutstate=, removeCutCells=False)` | `FilteredGrid` | From precomputed cut state. |
| `makeFilteredGrid(grid, mask=)` | `FilteredGrid` | Boolean mask, one entry per base cell. |
| `cutstate(grid, domain, nseedpoints=4, scaling=)` | — | Classify cells inside/outside/cut. |
| `gridsplit(lengths, targetNumber)` | — | Partition helper. |
| `makeUnstructuredMesh(...)` | mesh | Simplex/unstructured. |

Grid/mesh methods: `.ncells()`, `.nleaves()`, `.ndim`, `.boundingBox()`, `.memoryUsage()`, `.refine(strategy)` or `.refine(leafIndices)`, `.baseGrid()`, `.refinementLevels(fullHierarchy=False)`, `.shape(axis)` (CartesianGrid), `print(mesh)`.

### Refinement strategies (passed to `.refine(...)`)

`refineTowardsBoundary(domain, depth)`, `refineInsideDomain(...)`, `refineWithLevelFunction(...)`, `refineAdaptively(oldMesh, refineFlags)`, `refinementOr(...)`. `refineFlags` is a per-cell int list (`-1` coarsen / `0` keep / `1` refine — see adaptive example).

### Implicit geometry (CSG) — for the finite cell method

Primitives (all dimension-inferred from the point arguments):
`implicitSphere(center, radius)`, `implicitCube(x1, x2)`, `implicitEllipsoid(origin, radii)`, `implicitHalfspace(origin, normal)`, `implicitThreshold(scalarField, threshold)`, `implicitTransformation(func, transform)`.

Boolean ops: `implicitUnion([...])`, `implicitIntersection([...])`, `implicitSubtraction([...])`, `invert(func)`. Also `extrude(func, minValue, maxValue, axis)`.

Transforms: `translation(...)`, `scaling(...)`, `rotation(phi)` (2D) / `rotation(normal, phi)` (3D), `concatenate(...)`. A transform is callable on point lists (e.g. `rotation((30°))([(0,1),(0.5,1)])`).

Implicit-function methods: `.asfield(value0=0.0, value1=1.0)` → scalar field (outside/inside values), `func(xyz)` → bool. `implicitFunction(ndim, func)` is a Python helper = `implicitThreshold(scalarField(ndim, func), 0.5)`.

### Bases

| Function | Space |
|----------|-------|
| `makeHpTrunkSpace(grid, degree=1, nfields=1)` | Multi-level *hp* **trunk** space (`MultilevelHpBasis`). |
| `makeHpTensorSpace(grid, degree=1, nfields=1)` | Multi-level *hp* **tensor-product** space. |
| `makeHpTrunkSpace(grid, grading, nfields=1)` | Trunk space with per-level degree grading. |
| `makeBSplineBasis(grid, degree, continuity=None, nfields=1)` | B-spline basis. |
| `makeUnstructuredBasis(mesh, nfields=1)` | On unstructured mesh. |
| `makeFilteredBasis(...)` | Restrict an existing basis. |
| `makeDummyBasis(mesh, nfields=1)` | No DOFs — geometry-only postprocessing. |

Gradings: `UniformGrading`, `LinearGrading`, `InterpolatedGrading`. Degree factories: `makeHpTrunkSpaceFactory`, `makeHpTensorSpaceFactory`. `countTrunkSpaceDofs(...)`.

Basis methods: `.ndof()` (**no trailing s**), `.nelements()`, `.nfields()`, `.ndim`, `.locationMaps()`, `.maxdegree()`, `.memoryUsage()`, `print(basis)`.

Polynomials: `integratedLegendrePolynomials`, `equidistantLagrangePolynomials`, `gaussLobattoLagrangePolynomials`. Evaluators: `scalarEvaluator(basis, dofs, ifield, diffOrder, icomponent)`, `vectorEvaluator(basis, dofs, diffOrder)`, `evaluatorComponent(...)`, `findSupportElements(...)`, `findSupportedDofs(...)`.

### Fields

```python
mlhp.scalarField(ndim, value_or_expr)          # constant, or expression string
mlhp.vectorField(idim, list_or_expr, odim=None)
```

- Constant: `mlhp.scalarField(3, 1.5)`, `mlhp.vectorField(3, [0.0, 0.0, -9.81])`.
- **Expression strings** are parsed from Python syntax over variables `x,y,z` (or `r,s,t`, or subscripts `[0],[1],[2]`): e.g. `mlhp.scalarField(2, "sin(x) * y + 1")`, `mlhp.vectorField(2, "[ -y, x ]")`. Supports arithmetic, comparisons, boolean ops, and `a if cond else b` (lowers to `select(cond, a, b)`).
- From a compiled C-function pointer: pass `address=` (or a `func` with `.address`); requires `odim=` for vector fields. Used with numba `cfunc`.

Other field builders: `scalarFieldFromVoxelData(data, nvoxels, lengths)` (voxel image → field; `data` is a `FloatVector`/`DoubleVector`), `selectScalarField(domains, default=None)`, `sliceLast(function, value=0.0)`, `expandDimension(function, index=D)`.

### Quadrature

| Function | Use |
|----------|-----|
| `gridQuadrature(nsubcells=[1]*D)` | Tensor Gauss; `[1]*D` = one rule per element (preintegrated). |
| `standardQuadrature(ndim, ...)` | Standard Gauss rule. |
| `spaceTreeQuadrature(domain, depth, epsilon)` | FCM space-tree partitioning. |
| `momentFittingQuadrature(domain, depth, epsilon, nseedpoints=)` | FCM moment fitting (accurate cut cells). |
| `simplexQuadrature(*intersectWithMesh(surfMesh, mesh))` | Integrate over an embedded surface mesh. |
| `quadratureOnMeshFaces(mesh, faces)` | Surface quadrature on box faces (for Neumann). |
| `meshProjectionQuadrature`, `cellMeshQuadrature`, `cachedQuadrature` | Specialized. |
| `gaussLegendreRule`, `gaussLobattoRule` | 1D rules. |

Order control: `absoluteQuadratureOrder(...)`, `relativeQuadratureOrder(ndim, offset)`.

### Domain integrands (physics)

```python
mlhp.smallStrainKinematics(ndim)                       # KinematicEquation
mlhp.isotropicElasticMaterial(E_field, nu_field)       # 3D
mlhp.planeStressMaterial(E_field, nu_field)            # 2D plane stress
mlhp.planeStrainMaterial(E_field, nu_field)            # 2D plane strain
mlhp.staticDomainIntegrand(kinematics, material, rhs_vectorField)
mlhp.internalEnergyIntegrand(dofs, kinematics, material)
```

Other prebuilt integrands: `poissonIntegrand(...)`, `transientPoissonIntegrand(...)`, `l2DomainIntegrand(...)`, `functionIntegrand(...)`, `l2ErrorIntegrand(dofs, ...)`, `energyErrorIntegrand(dofs, ...)`, `l2BasisProjectionIntegrand(...)`.

**Custom compiled integrand** (numba/cffi): `mlhp.domainIntegrand(ndim, callback, types=[...], maxdiff, tmpdofs=0)` where `callback` is a numba `cfunc` with a fixed signature (element matrix/vector assembly in C) — see `examples/poisson_compiled.py`. `types` is a list of `mlhp.AssemblyType` (`Scalar`, `Vector`, `UnsymmetricMatrix`, `SymmetricMatrix`).

### Surface integrands (Neumann / weak BCs)

`neumannIntegrand(vectorField)` (traction vector), `normalNeumannIntegrand(scalarField)` (pressure along outward normal), `l2BoundaryIntegrand(penalty, value)` (penalty Dirichlet), `l2NormalIntegrand(...)`, `nitscheIntegrand(...)`, `reactionForceIntegrand(...)`, `normalDotProductIntegrand(...)`, `functionSurfaceIntegrand(...)`.

### Boundary conditions

```python
# Vector field → constrains ALL components on the faces:
dirichlet = mlhp.integrateDirichletDofs(mlhp.vectorField(D, [0.0]*D), basis, [0])

# Scalar field + ifield → constrains ONE component (roller / normal-only):
bc = mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [face], ifield=face//2)

dirichlet = mlhp.combineDirichletDofs([bc0, bc1, ...])   # merge several
constrained_dofs = np.array(dirichlet[0])                # constrained DOF indices
all_dofs = mlhp.inflateDofs(interior_dofs, dirichlet)    # reinsert BC values after solve
```

`dirichlet` is `[indices, values]`. `dirichlet[0]` is what `allocateSparseMatrix` needs.

### Assembly

```python
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])   # condenses constrained DOFs
vector = mlhp.allocateRhsVector(matrix)
mlhp.integrateOnDomain(basis, integrand, [matrix, vector],
                       quadrature=mlhp.gridQuadrature([1]*D), dirichletDofs=dirichlet)
mlhp.integrateOnSurface(basis, neumann, [vector], faceQuad, dirichletDofs=dirichlet)
```

Targets list can be any subset/order of `[matrix, vector]`, `[vector]`, or `[ScalarDouble]` (for energy). `projectOnto(...)` for L2 projection.

### Linear algebra & solvers

Solvers (Python wrappers in `mlhp.py`):
```python
x = mlhp.cg(A, b, x0=None, *, rtol=1e-10, atol=0.0, maxiter=None, M=None,
            residualNorms=False, tmp=None)
x = mlhp.bicgstab(A, b, ...)            # same signature
# residualNorms=True → returns (x, norms); A may be a matrix or a linearOperator
solve = mlhp.makeCGSolver(rtol=1e-10, ...)   # returns a callable solve(A,b) with diag precond
```

Preconditioners: `diagonalPreconditioner(matrix)`, `additiveSchwarzPreconditioner(matrix, basis, dirichlet[0])`, `noPreconditioner()`. `linearOperator(matrix)` wraps a matrix as an operator.

Sparse matrix (`SymmetricSparseMatrix` / `UnsymmetricSparseMatrix`) members:
`.nnz`, `.shape`, `.symmetricHalf`, `.todense()`, `.multiply(v)` / `M * v`, `.memoryUsage()`, `.copy()`, and CSR exports `.csr_arrays`, `.indptr_array`, `.indices_array`, `.data_array` (plus `_list`/`_buffer`/`_address` variants). **Bridge to scipy**: `scipy.sparse.csr_matrix(*matrix.csr_arrays)`.

Vectors — `DoubleVector` / `FloatVector`:
`DoubleVector(size)`, `DoubleVector(size, value)`, `DoubleVector(list)`; `.array` (numpy view), `.tolist()`, `.size`, `.shape`, `.copy()`, `.resize(n)`, `len(v)`, `v[i]`, `.address`, `.buffer`. Build from numpy: `mlhp.DoubleVector(arr.tolist())`. `ScalarDouble(x)` with `.get()` (accumulator target). Helpers: `norm(v)`, `copy(vector=, scaling=, offset=)`, `fill(v, value)`, `add(...)`, `split(...)`, `filterZeros(matrix, threshold=0.0)`, `makeScalars(n, value=0.0)`.

### Postprocessing

Processors (passed as a list, order defines output channel order):
```python
mlhp.solutionProcessor(ndim, dofs, "Displacement")
mlhp.functionProcessor(field_or_domain, "Indicator")
mlhp.vonMisesProcessor(dofs, kinematics, material)
mlhp.stressProcessor(...)  mlhp.strainProcessor(...)  mlhp.strainEnergyProcessor(...)
mlhp.cellIndexProcessor(ndim)   mlhp.cellDataProcessor(ndim, ...)   mlhp.kdTreeInfoProcessor(ndim)
```

Postprocessing meshes (sampling resolution): `gridCellMesh(resolution, topologies=PostprocessTopologies.Volumes)`, `domainCellMesh(domain, resolution)`, `boundaryCellMesh(...)`, `quadraturePointCellMesh(...)`, `customCellMesh(...)`, `degreeOffsetResolution(basis)`. `PostprocessTopologies`: `Nothing|Corners|Edges|Faces|Volumes` (combine with `|`). `CellType`: `NCube|Simplex`.

Outputs / writers:
```python
out = mlhp.PVtuOutput(filename="result")     # parallel VTU (.pvtu); also VtuOutput
mlhp.basisOutput(basis, postmesh, out, processors)
# in-memory for plotting:
acc = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, acc, processors)
ux = np.array(acc.data()[0])[0::D]           # channel 0 = first processor (interleaved)
tri = acc.triangulation()                    # matplotlib Triangulation (2D helper)
```
`basisOutput` may be called repeatedly on the same processors (once per writer). `meshOutput(mesh, ...)` for geometry-only.

Error indicators / recovery: `stressJumpIndicator(basis, dofs, kinematics, material, scaling=, order=)`, `stressDivergenceIndicator(...)`, `projectGradient(...)`.

### Triangulation / STL / embedded surfaces

`simplexMesh(vertices, cells)`; helpers `triangulation(vertices, triangles)` (3D) and `lineSegments(vertices, segments)` (2D). `readStl(filename, correctOrdering=False)`, `.writeStl(...)`, `.writeVtu(...)`. `intersectWithMesh(surfMesh, mesh)`, `rayIntersectionDomain(...)`, `recoverDomainBoundary(mesh, ...)`, `buildKdTree(...)`. SimplexMesh methods: `.ncells()`, `.nvertices()`, `.cellVertices(i)`, `.cellNormal(i)`, `.boundingBox()`, `.measure()` (`.area()`/`.length()`), `.transform(t)`, `.filter(func)`, and 2D `.plot(axis=, color=)`.

Analytic solutions for verification: `makeSingularSolution(ndim)`, `makeAmLinearSolution(...)`.

---

## Recipe: finite cell method (immersed boundary)

Embed CSG geometry in a structured grid; integrate cut cells with moment fitting:

```python
domain = mlhp.invert(mlhp.implicitCube([0,0.1,0.1], [1,0.9,0.9]))     # void inside
mesh = mlhp.makeRefinedGrid(nelements, lengths)
mesh.refine(mlhp.refineTowardsBoundary(domain, refinementDepth))
basis = mlhp.makeHpTensorSpace(mesh, degree, nfields=D)
quadrature = mlhp.momentFittingQuadrature(domain, depth=degree+1, epsilon=1e-3)
mlhp.integrateOnDomain(basis, integrand, [matrix, vector],
                       dirichletDofs=dirichlet, quadrature=quadrature)
# weak Dirichlet on a cut surface: penalty l2BoundaryIntegrand on simplexQuadrature(intersectWithMesh(...))
```

`epsilon` is the fictitious-domain stiffness scaling (`alphaFCM`); smaller = more accurate cut but worse conditioning (needs `additiveSchwarzPreconditioner`). `functionProcessor(domain)` adds an in/out indicator to the output.

---

## Recipe: voxel FEM (`projects/16_elastic_fem/`)

These drivers solve linear elasticity on a CT-derived voxel image, with three solver backends:
- `elastic_mlhp.py` — assembled sparse matrix + `mlhp.cg`
- `elastic_mlhp_matrixfree.py` — matrix-free matvec (`K_ref` + location maps) + scipy CG
- `elastic_mlhp_cuda.py` — matrix-free CUDA matvec (CuPy `RawModule`) + cupyx CG

CT geometry (`create_CT_*.py`) is run once → `.npz` in `code/data/`; drivers load from there.

### CT indicator → material field

Indicators are stored as `uint8` (0 = void, 255 = solid). **Never store a float32 copy** — derive on the fly. `ALPHA = 1e-5` is the minimum E scaling for void voxels (fictitious-domain stiffness).

```python
indicator = ct["indicator"]                     # uint8, (Nx,Ny[,Nz])
ncells  = list(indicator.shape)
lengths = [float(ct["Lx"]), float(ct["Ly"])] + ([float(ct["Lz"])] if D==3 else [])
E_values = E * np.maximum(indicator.ravel("C") / 255.0, ALPHA)     # float64, per element

# For an mlhp material field (assembled driver):
E_vec = mlhp.FloatVector(E * np.maximum(indicator.ravel("C").astype(np.float32)/255.0, ALPHA))
E_field = mlhp.scalarFieldFromVoxelData(E_vec, nvoxels=ncells, lengths=lengths)
nu_field = mlhp.scalarField(D, NU)
material = mlhp.planeStressMaterial(E_field, nu_field) if D==2 else mlhp.isotropicElasticMaterial(E_field, nu_field)
```

`indicator.ravel("C")` (shape `(Nx,Ny[,Nz])`) matches mlhp element ordering — see numbering section. `FloatVector` and `DoubleVector` both work with `scalarFieldFromVoxelData`. Use `mlhp.gridQuadrature([1]*D)` (one sub-cell = constant E per element = preintegrated).

### Roller BCs + Neumann traction (uni-axial tension)

```python
bc_faces = [0, 2] if D == 2 else [0, 2, 4]      # x-, y-, (z-)
dirichlet = mlhp.combineDirichletDofs([
    mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [f], ifield=f//2)
    for f in bc_faces])
constrained_dofs = np.array(dirichlet[0])

traction = FORCE / Ly if D == 2 else FORCE / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])     # face 1 = x+
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)
```

### Matrix-free: K_ref + location maps

Every element stiffness is `E_i * K_ref` (identical geometry). Extract `K_ref` by assembling **one** unit-E element with empty BCs (full `ndof_e × ndof_e`):

```python
mesh1  = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1]*D, lengths=elem_lengths))
basis1 = mlhp.makeHpTrunkSpace(mesh1, degree=DEGREE, nfields=D)
c_ref  = (mlhp.planeStressMaterial if D==2 else mlhp.isotropicElasticMaterial)(mlhp.scalarField(D,1.0), nu_field)
i_ref  = mlhp.staticDomainIntegrand(mlhp.smallStrainKinematics(D), c_ref, mlhp.vectorField(D,[0.0]*D))
de = mlhp.combineDirichletDofs([])                       # no condensation
m_ref = mlhp.allocateSparseMatrix(basis1, de[0]); v_ref = mlhp.allocateRhsVector(m_ref)
mlhp.integrateOnDomain(basis1, i_ref, [m_ref, v_ref], quadrature=mlhp.gridQuadrature([1]*D), dirichletDofs=de)
K_ref = np.array(m_ref.todense())
efts  = np.array(basis.locationMaps())                   # (n_elem, ndof_e)

def matvec(u):
    Ku = E_values[:, None] * (u[efts] @ K_ref.T)         # per-element scatter
    r = np.zeros(ndof); np.add.at(r, efts, Ku)
    r[constrained_dofs] = u[constrained_dofs]            # identity rows for BC dofs
    return r
```

**RHS for the matrix-free path**: with zero body force the domain integral contributes nothing to `vector` — only the Neumann surface term does — so assemble the RHS with a **dummy E=1 material** (the real `E_field` is not needed) and discard the matrix:
```python
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0]); vector = mlhp.allocateRhsVector(matrix)
mlhp.integrateOnDomain(basis, i_rhs_dummy, [matrix, vector], quadrature=mlhp.gridQuadrature([1]*D), dirichletDofs=dirichlet)
# ... add Neumann ...
rhs = np.zeros(ndof); rhs[~np.isin(np.arange(ndof), constrained_dofs)] = np.array(list(vector))
del matrix, vector
```
Diagonal preconditioner: `np.add.at(diag, efts, E_values[:,None]*np.diag(K_ref)[None,:]); diag[constrained_dofs]=1.0`. Solve with `scipy.sparse.linalg.cg(LinearOperator(...), rhs, M=LinearOperator(matvec=lambda v: v/diag), rtol=1e-10)`.

### CUDA matvec (CuPy)

Kernel in `elasticity_mf_mlhp.cu`, loaded via `cp.RawModule(code=...)`. `efts` must be `cp.int32` (kernel takes `const int*`); `K_ref` passed flat as `K_ref.ravel("C")`. **cupyx CG uses `tol=`, not `rtol=`.** Pattern: `kernel_matvec((grid,),(block,),(u,Ku,E_values,efts,K_ref,n_elem,ndof_e))`, force BC rows to identity, wrap in `cupyx.scipy.sparse.linalg.LinearOperator`, solve with `cupyx.scipy.sparse.linalg.cg(..., tol=1e-10)`.

### 2D matplotlib output

```python
all_dofs = mlhp.DoubleVector(sol.tolist())               # scipy/cupyx return numpy
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(indicator.ravel("C").astype(np.float32)/255.0), nvoxels=ncells, lengths=lengths)
processors = [mlhp.solutionProcessor(D, all_dofs, "Displacement"),
              mlhp.functionProcessor(indicator_field, "Indicator")]
postmesh = mlhp.gridCellMesh([DEGREE+2]*D)

acc = mlhp.DataAccumulator(); mlhp.basisOutput(basis, postmesh, acc, processors)
ux = np.array(acc.data()[0])[0::2]; uy = np.array(acc.data()[0])[1::2]
ind = np.array(acc.data()[1])
tri = acc.triangulation(); tri.set_mask(ind[tri.triangles].mean(axis=1) < 0.5)
ax.tricontourf(tri, ux, cmap="turbo", levels=24)         # elasticity → turbo colormap
```

---

## Common pitfalls

| Mistake | Correct |
|---------|---------|
| `basis.ndofs()` | `basis.ndof()` (no trailing s) |
| Mixing objects of different `D` | All meshes/bases/fields must share one dimension |
| `cupyx ...cg(..., rtol=...)` | cupyx uses `tol=`; only `mlhp.cg`/scipy use `rtol=` |
| Vector `integrateDirichletDofs` for a roller BC | Use the **scalar** overload with `ifield=` to constrain one component |
| Using real `E_field` in matrix-free RHS assembly | Not needed — dummy `E=1`; zero body force ⇒ domain integral adds nothing to the vector |
| `DATA_DIR / (args.ct or default)` then `np.load` | `DATA_DIR / (args.ct if args.ct else default)` — `/ None` crashes |
| Storing the indicator as float32 | Keep `uint8`; derive `E_values = E*np.maximum(indicator.ravel("C")/255.0, ALPHA)` |
| `FloatVector(indicator.ravel("C"))` without `/255` | uint8 is 0–255; divide before passing |
| `efts` as int64 for the CUDA kernel | Must be `cp.int32` to match `const int*` |
| `K_ref` passed 2D to the kernel | Pass `K_ref.ravel("C")` (flat row-major) |
| Scipy from an mlhp matrix by hand | `scipy.sparse.csr_matrix(*matrix.csr_arrays)` |
| Forgetting `inflateDofs` after solve | Interior solution omits BC DOFs; `inflateDofs(interior, dirichlet)` restores full vector |
