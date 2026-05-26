---
name: mlhp_voxel_fem
description: Reference for mlhp-based voxel FEM drivers in this repo. Covers API for mesh/basis, material fields, assembly, BCs, solvers (sparse, matrix-free NumPy, matrix-free CUDA), and postprocessing. Invoke before writing or editing any `projects/16_elastic_fem/` file, or any new mlhp driver.
---

This skill captures the mlhp API as used in `projects/16_elastic_fem/`. The three reference drivers are:
- `elastic_mlhp.py` — assembled sparse matrix + `mlhp.cg`
- `elastic_mlhp_matrixfree.py` — matrix-free matvec + scipy CG
- `elastic_mlhp_cuda.py` — matrix-free CUDA matvec + cupyx CG

CT geometry files (`create_CT_2D.py` / `create_CT_3D.py`) are run once and produce `.npz` files in `code/data/`. All drivers load from those files.

---

## Package

```python
import mlhp
```

Installed as `pip install mlhp` (pymlhpcore wheel). The `solvers/mlhp` submodule is the pinned C++ source (0.1.2); the pip package is the primary import.

---

## Mesh and basis

```python
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=ncells, lengths=lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
ndof = basis.ndof()       # total DOFs (no trailing 's')
print(basis)              # summary string
```

- `ncells`: `[Nx, Ny]` (2D) or `[Nx, Ny, Nz]` (3D).
- `lengths`: `[Lx, Ly]` or `[Lx, Ly, Lz]` — physical domain size.
- `nfields=D` — one field per spatial dimension (displacement vector).
- `degree=1` → Q1; `degree=2` → Q2; etc.

---

## CT geometry — uint8 storage

Indicators are saved as `uint8` (0 = void, 255 = solid). **Never store a float32 copy** — convert on the fly where needed.

**Load in a driver**:
```python
DATA_DIR = BASE_DIR / "../../data"
ALPHA = 1e-5   # minimum E scaling for void voxels

ct_file = args.ct if args.ct else f"plate_hole_{D}D.npz"
ct = np.load(DATA_DIR / ct_file)
indicator = ct["indicator"]   # uint8, shape (Nx, Ny) or (Nx, Ny, Nz)

if D == 2:
    Lx, Ly = float(ct["Lx"]), float(ct["Ly"])
    Nx, Ny = indicator.shape
    ncells = [Nx, Ny]
    lengths = [Lx, Ly]
    elem_lengths = [Lx / Nx, Ly / Ny]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    ncells = [Nx, Ny, Nz]
    lengths = [Lx, Ly, Lz]
    elem_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]
```

**Derive E_values** (float64, computed once from uint8):
```python
E_values = E * np.maximum(indicator.ravel("C") / 255.0, ALPHA)
```

**For mlhp material field** (assembled sparse driver only):
```python
E_vec = mlhp.FloatVector((E * np.maximum(indicator.ravel("C").astype(np.float32) / 255.0, ALPHA)))
E_field = mlhp.scalarFieldFromVoxelData(E_vec, nvoxels=ncells, lengths=lengths)
```
`FloatVector` works with `scalarFieldFromVoxelData` (verified). `DoubleVector` also works.

**For indicator postprocessing field** (convert on the fly):
```python
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(indicator.ravel("C").astype(np.float32) / 255.0),
    nvoxels=ncells, lengths=lengths)
```

**Voxel ordering**: `indicator.ravel("C")` with shape `(Nx, Ny[, Nz])` matches `basis.locationMaps()` element ordering for structured grids — verified against mlhp's `findVoxel` in `spatial.cpp`.

---

## Constitutive models

```python
nu_field = mlhp.scalarField(D, NU)

# 2D plane stress
material = mlhp.planeStressMaterial(E_field, nu_field)

# 3D isotropic linear elastic
material = mlhp.isotropicElasticMaterial(E_field, nu_field)
```

---

## Kinematics and integrand

```python
kinematics = mlhp.smallStrainKinematics(D)
integrand = mlhp.staticDomainIntegrand(kinematics, material, mlhp.vectorField(D, [0.0] * D))
```

**Body force**: replace the zero vector to add a distributed load — the constitutive model does not affect the body force term in the RHS, only the stiffness matrix:
```python
integrand = mlhp.staticDomainIntegrand(kinematics, material, mlhp.vectorField(D, [0.0, -RHO * G]))
```

---

## Quadrature — preintegrated voxel FEM

```python
quadrature = mlhp.gridQuadrature(nsubcells=[1] * D)
```

One sub-cell per element = E constant per element = preintegrated case.

---

## Dirichlet BCs

Face IDs (same in 2D and 3D):
| ID | Face |
|----|------|
| 0 | x− (left) |
| 1 | x+ (right) |
| 2 | y− (bottom) |
| 3 | y+ (top) |
| 4 | z− (back) |
| 5 | z+ (front) |

**Uni-axial tension** — each face constrains only its own normal component:
```python
bc_faces = [0, 2] if D == 2 else [0, 2, 4]
bc_list = [
    mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2)
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)
constrained_dofs = np.array(dirichlet[0])
```

`ifield=face // 2` → face 0 → ux=0, face 2 → uy=0, face 4 → uz=0.

**Fully clamped face**:
```python
bc_list = [
    mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [0], ifield=field)
    for field in range(D)
]
```

---

## Assembly — assembled sparse driver

```python
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
mlhp.integrateOnDomain(basis, integrand, [matrix, vector],
                       quadrature=quadrature, dirichletDofs=dirichlet)
```

---

## Neumann traction (normal, on a face)

```python
traction = FORCE / Ly if D == 2 else FORCE / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])   # face 1 = x+
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)
```

---

## Solve — assembled sparse matrix

```python
P = mlhp.diagonalPreconditioner(matrix)
interior_dofs, residuals = mlhp.cg(matrix, vector, M=P,
                                   rtol=1e-10, maxiter=20000, residualNorms=True)
all_dofs = mlhp.inflateDofs(interior_dofs, dirichlet)
print(f"CG: {len(residuals)} iterations")
print(f"max displacement: {max(abs(v) for v in all_dofs):.3e}")
```

---

## Matrix-free approach — K_ref + locationMaps

Every element's stiffness is `E_i * K_ref`. The matvec is a batched scatter-add; K is never assembled globally.

### Extract K_ref (one-element assembly)

```python
kinematics = mlhp.smallStrainKinematics(D)
mesh1 = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
basis1 = mlhp.makeHpTrunkSpace(mesh1, degree=DEGREE, nfields=D)
c_ref = (mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field) if D == 2
         else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field))
i_ref = mlhp.staticDomainIntegrand(kinematics, c_ref, mlhp.vectorField(D, [0.0] * D))
de = mlhp.combineDirichletDofs([])   # empty BCs — no condensation, full ndof_e × ndof_e
m_ref = mlhp.allocateSparseMatrix(basis1, de[0])
v_ref = mlhp.allocateRhsVector(m_ref)
mlhp.integrateOnDomain(basis1, i_ref, [m_ref, v_ref],
                       quadrature=mlhp.gridQuadrature(nsubcells=[1] * D), dirichletDofs=de)
K_ref = np.array(m_ref.todense())   # (ndof_e, ndof_e)
K_ref_diag = np.diag(K_ref)
```

### RHS assembly — dummy material is fine

With zero body force the domain integral contributes nothing to `vector` — only the Neumann surface integral does. Therefore `E_field` is **not needed**: use a dummy `E=1` material. The assembled `matrix` is discarded immediately.

```python
c_rhs = (mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field) if D == 2
         else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field))
i_rhs = mlhp.staticDomainIntegrand(kinematics, c_rhs, mlhp.vectorField(D, [0.0] * D))
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
mlhp.integrateOnDomain(basis, i_rhs, [matrix, vector],
                       quadrature=mlhp.gridQuadrature(nsubcells=[1] * D), dirichletDofs=dirichlet)
# add Neumann traction ...
del matrix, vector
```

If a non-zero body force is added, `c_rhs` still doesn't need to match the real E distribution — the body force term `∫ N^T f dΩ` does not involve the constitutive model.

### Inflate RHS to full DOF space

```python
interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained_dofs] = False
rhs = np.zeros(ndof)
rhs[np.where(interior_mask)[0]] = np.array(list(vector))
del matrix, vector
```

### Element freedom tables

```python
efts = np.array(basis.locationMaps())   # (n_elem, ndof_e)
```

### NumPy matvec

```python
def matvec(u):
    Ku_local = E_values[:, None] * (u[efts] @ K_ref.T)
    result = np.zeros(ndof)
    np.add.at(result, efts, Ku_local)
    result[constrained_dofs] = u[constrained_dofs]
    return result
```

### Diagonal preconditioner (matrix-free)

```python
diag = np.zeros(ndof)
np.add.at(diag, efts, E_values[:, None] * K_ref_diag[None, :])
diag[constrained_dofs] = 1.0
```

### scipy CG (full-space)

```python
A_op = scipy.sparse.linalg.LinearOperator((ndof, ndof), matvec=matvec)
P_op = scipy.sparse.linalg.LinearOperator((ndof, ndof), matvec=lambda v: v / diag)

iters = [0]
def callback(x): iters[0] += 1

sol, info = scipy.sparse.linalg.cg(A_op, rhs, M=P_op, rtol=1e-10, maxiter=20000, callback=callback)
```

---

## CUDA matvec — CuPy RawModule

Kernel file: `elasticity_mf_mlhp.cu`.

```python
cuda_source = (BASE_DIR / "elasticity_mf_mlhp.cu").read_text()
module = cp.RawModule(code=cuda_source)
kernel_matvec = module.get_function("cuda_matvec")
kernel_k_diag = module.get_function("cuda_k_diag")

n_elem = len(E_values)
ndof_e = K_ref.shape[0]
block = 256
grid = (n_elem + block - 1) // block

K_ref_gpu = cp.array(K_ref.ravel("C"), dtype=cp.float64)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
E_values_gpu = cp.array(E_values, dtype=cp.float64)
rhs_gpu = cp.array(rhs, dtype=cp.float64)
constrained_gpu = cp.array(constrained_dofs, dtype=cp.int32)

K_diag_gpu = cp.zeros(ndof, dtype=cp.float64)
kernel_k_diag((grid,), (block,), (K_diag_gpu, E_values_gpu, efts_gpu, K_ref_gpu, n_elem, ndof_e))
K_diag_gpu[constrained_gpu] = 1.0

def matvec_gpu(u_gpu):
    Ku_gpu = cp.zeros(ndof, dtype=cp.float64)
    kernel_matvec((grid,), (block,), (u_gpu, Ku_gpu, E_values_gpu, efts_gpu, K_ref_gpu, n_elem, ndof_e))
    Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
    return Ku_gpu

A_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=matvec_gpu)
P_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

cp.cuda.Stream.null.synchronize()
sol_gpu, info = cp_splinalg.cg(A_op, rhs_gpu, M=P_op, tol=1e-10, maxiter=20000, callback=callback)
cp.cuda.Stream.null.synchronize()
sol = sol_gpu.get()
```

cupyx CG uses `tol=` not `rtol=`.

---

## Postprocessing

### Processors

```python
all_dofs = mlhp.DoubleVector(sol.tolist())   # scipy/cupyx: sol is numpy array
# or for mlhp.cg: all_dofs = mlhp.inflateDofs(interior_dofs, dirichlet)

indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(indicator.ravel("C").astype(np.float32) / 255.0),
    nvoxels=ncells, lengths=lengths)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)
```

### PVTU export (2D and 3D)

```python
RESULTS_DIR = BASE_DIR / "../../results"

out = str(RESULTS_DIR / f"elastic_mlhp_{ct_file[:-4]}")
Path(out).parent.mkdir(parents=True, exist_ok=True)
output = mlhp.PVtuOutput(filename=out)
mlhp.basisOutput(basis, postmesh, output, processors)
print(f"VTU written to {out}.pvtu")
```

### 2D matplotlib (DataAccumulator)

`basisOutput` can be called multiple times on the same `processors` — once for PVTU, once for DataAccumulator.

```python
result = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, result, processors)

ux = np.array(result.data()[0])[0::2]   # interleaved [ux0,uy0,...] → ux
uy = np.array(result.data()[0])[1::2]   # → uy
ind_viz = np.array(result.data()[1])     # scalar indicator per point

tri = result.triangulation()
tri.set_mask(ind_viz[tri.triangles].mean(axis=1) < 0.5)

fig, ax = plt.subplots()
cb = ax.tricontourf(tri, ux, cmap="turbo", levels=24)
fig.colorbar(cb)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0)
plt.show()
```

`result.data()` order matches `processors` list:
- `[0]` → displacement (interleaved, length `D * npoints`)
- `[1]` → indicator (length `npoints`)

---

## Common pitfalls

| Mistake | Correct |
|---------|---------|
| `basis.ndofs()` | `basis.ndof()` (no trailing s) |
| `cp_splinalg.cg(..., rtol=...)` | `tol=` (cupyx doesn't accept `rtol`) |
| Using `E_field` in matrix-free RHS assembly | Not needed — use dummy `E=1`; body force = 0 means domain integral contributes nothing to vector |
| `ct = np.load(DATA_DIR / args.ct or ct_default)` | `DATA_DIR / None` crashes — use `DATA_DIR / args.ct if args.ct else ct_default` |
| Storing indicator as float32 | Keep as uint8; derive `E_values = E * np.maximum(indicator.ravel("C") / 255.0, ALPHA)` |
| `indicator.ravel("C")` without `/255` for FloatVector | uint8 values are 0–255; divide by 255 before passing to FloatVector |
| `efts_gpu` as int64 | Must be `cp.int32` to match `const int*` in kernel |
| `K_ref.ravel("C")` not passed | GPU kernel expects flat row-major array, not 2D |
