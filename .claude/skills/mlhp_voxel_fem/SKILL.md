---
name: mlhp_voxel_fem
description: Reference for mlhp-based voxel FEM drivers in this repo. Covers API for mesh/basis, material fields, assembly, BCs, solvers (sparse, matrix-free NumPy, matrix-free CUDA), and postprocessing. Invoke before writing or editing any `projects/16_elastic_fem/` file, or any new mlhp driver.
---

This skill captures the mlhp API as used in `projects/16_elastic_fem/`. The three reference drivers are:
- `2D_elastic_mlhp.py` — assembled sparse matrix + `mlhp.cg`
- `2D_elastic_mlhp_matrixfree.py` — matrix-free matvec + scipy CG
- `2D_elastic_mlhp_cuda.py` — matrix-free CUDA matvec + cupyx CG

CT geometry files are separate (`create_CT_2D.py` / `create_CT_3D.py`), loaded as `.npz` by all drivers.

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
basis = mlhp.makeHpTrunkSpace(mesh, degree=degree, nfields=D)
ndof = basis.ndof()       # total DOFs (no trailing 's')
print(basis)              # summary string
```

- `ncells`: `[Nx, Ny]` (2D) or `[Nx, Ny, Nz]` (3D).
- `lengths`: `[Lx, Ly]` or `[Lx, Ly, Lz]` — physical domain size.
- `nfields=D` — one field per spatial dimension (displacement vector).
- `degree=1` → Q1 (bilinear/trilinear); `degree=2` → Q2; etc.

---

## Spatially-varying material field from voxel data

```python
E_vec = mlhp.DoubleVector((E * indicator).ravel("C"))
E_field = mlhp.scalarFieldFromVoxelData(E_vec, nvoxels=ncells, lengths=lengths)
nu_field = mlhp.scalarField(D, nu)         # uniform scalar field
```

**Voxel ordering**: `indicator` must have shape `(Nx, Ny)` (2D) or `(Nx, Ny, Nz)` (3D), then `.ravel("C")` (last axis fastest). This matches mlhp's internal `findVoxel` in `spatial.cpp`. **Verified**: element ordering of `basis.locationMaps()` is consistent with C row-major ravel for structured grids.

The indicator array itself stores the E-scaling factor: `1.0` for solid, `1e-5` for void. For the material call, pass `E * indicator`.

---

## Constitutive models

```python
# 2D plane stress
material = mlhp.planeStressMaterial(E_field, nu_field)

# 3D isotropic linear elastic
material = mlhp.isotropicElasticMaterial(E_field, nu_field)
```

---

## Kinematics and integrand

```python
kinematics = mlhp.smallStrainKinematics(D)
body_force = mlhp.vectorField(D, [0.0] * D)
integrand = mlhp.staticDomainIntegrand(kinematics, material, body_force)
```

---

## Quadrature — preintegrated voxel FEM

```python
quadrature = mlhp.gridQuadrature(nsubcells=[1] * D)
```

One sub-cell per element. Because element resolution = voxel resolution, E is constant per element — this is the preintegrated case. mlhp automatically selects `degree+1` Gauss points per direction.

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

**Uni-axial tension** (correct BCs — each face constrains only its own normal component):
```python
bc_faces = [0, 2] if D == 2 else [0, 2, 4]
bc_list = [
    mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2)
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)
constrained = np.array(dirichlet[0])
```

`ifield=face // 2` → face 0 → ux=0, face 2 → uy=0, face 4 → uz=0. Tangential directions are free (Poisson contraction allowed).

**Fully clamped face** (all components fixed on one face):
```python
bc_list = [
    mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [0], ifield=field)
    for field in range(D)
]
```

---

## Assembly

```python
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

mlhp.integrateOnDomain(basis, integrand, [matrix, vector],
                       quadrature=quadrature, dirichletDofs=dirichlet)
```

---

## Neumann traction (normal, on a face)

```python
traction = force / Ly if D == 2 else force / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])   # face 1 = x+ (right)
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

The core idea: every element's stiffness is `E_i * K_ref`, where `K_ref` is the stiffness of a single reference element with `E=1`. The matvec is then a batched scatter-add.

### Extract K_ref (one-element assembly)

```python
elem_lengths = [Lx / Nx, Ly / Ny] if D == 2 else [Lx / Nx, Ly / Ny, Lz / Nz]

mesh1 = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
basis1 = mlhp.makeHpTrunkSpace(mesh1, degree=degree, nfields=D)
c_ref = (mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field) if D == 2
         else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field))
i_ref = mlhp.staticDomainIntegrand(kinematics, c_ref, mlhp.vectorField(D, [0.0] * D))
de = mlhp.combineDirichletDofs([])
m_ref = mlhp.allocateSparseMatrix(basis1, de[0])
v_ref = mlhp.allocateRhsVector(m_ref)
mlhp.integrateOnDomain(basis1, i_ref, [m_ref, v_ref],
                       quadrature=mlhp.gridQuadrature(nsubcells=[1] * D), dirichletDofs=de)
K_ref = np.array(m_ref.todense())   # (ndof_e, ndof_e)
K_ref_diag = np.diag(K_ref)
```

DOF ordering in `K_ref` is consistent with `basis.locationMaps()` because both use the same mlhp basis.

### Element freedom tables

```python
efts = np.array(basis.locationMaps())   # (n_elem, ndof_e)
```

### Inflate RHS to full DOF space

mlhp's `mlhp.cg` operates on interior DOFs only. For scipy/cupyx CG (full-space), inflate first:

```python
interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained] = False
rhs = np.zeros(ndof)
rhs[np.where(interior_mask)[0]] = np.array(list(vector))
del matrix, vector
```

### NumPy matvec

```python
def matvec(u):
    u_local = u[efts]                              # (n_elem, ndof_e)
    Ku_local = E_values[:, None] * (u_local @ K_ref.T)  # (n_elem, ndof_e)
    result = np.zeros(ndof)
    np.add.at(result, efts, Ku_local)
    result[constrained] = u[constrained]           # identity on Dirichlet DOFs
    return result
```

### Diagonal preconditioner (matrix-free)

```python
diag = np.zeros(ndof)
np.add.at(diag, efts, E_values[:, None] * K_ref_diag[None, :])
diag[constrained] = 1.0
```

### scipy CG (full-space)

```python
import scipy.sparse.linalg

A_op = scipy.sparse.linalg.LinearOperator((ndof, ndof), matvec=matvec)
P_op = scipy.sparse.linalg.LinearOperator((ndof, ndof), matvec=lambda v: v / diag)

iters = [0]
def callback(x): iters[0] += 1

sol, info = scipy.sparse.linalg.cg(A_op, rhs, M=P_op, rtol=1e-10, maxiter=20000, callback=callback)
```

---

## CUDA matvec — CuPy RawModule

Kernel file: `elasticity_mf_mlhp.cu`. Load and compile at startup:

```python
import cupy as cp
import cupyx.scipy.sparse.linalg as cp_splinalg

cuda_source = (BASE_DIR / "elasticity_mf_mlhp.cu").read_text()
module = cp.RawModule(code=cuda_source)
kernel_matvec = module.get_function("cuda_matvec")
kernel_k_diag = module.get_function("cuda_k_diag")
```

Upload to GPU:

```python
n_elem = len(E_values)
ndof_e = K_ref.shape[0]
block = 256
grid = (n_elem + block - 1) // block

K_ref_gpu = cp.array(K_ref.ravel("C"), dtype=cp.float64)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
E_values_gpu = cp.array(E_values, dtype=cp.float64)
rhs_gpu = cp.array(rhs, dtype=cp.float64)
constrained_gpu = cp.array(constrained, dtype=cp.int32)
```

Diagonal on GPU:

```python
K_diag_gpu = cp.zeros(ndof, dtype=cp.float64)
kernel_k_diag((grid,), (block,),
              (K_diag_gpu, E_values_gpu, efts_gpu, K_ref_gpu, n_elem, ndof_e))
K_diag_gpu[constrained_gpu] = 1.0
```

GPU matvec:

```python
def matvec_gpu(u_gpu):
    Ku_gpu = cp.zeros(ndof, dtype=cp.float64)
    kernel_matvec((grid,), (block,),
                  (u_gpu, Ku_gpu, E_values_gpu, efts_gpu, K_ref_gpu, n_elem, ndof_e))
    Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
    return Ku_gpu
```

cupyx CG — use `tol=` not `rtol=`:

```python
A_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=matvec_gpu)
P_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

iters = [0]
def callback(x): iters[0] += 1

cp.cuda.Stream.null.synchronize()
sol_gpu, info = cp_splinalg.cg(A_op, rhs_gpu, M=P_op, tol=1e-10, maxiter=20000, callback=callback)
cp.cuda.Stream.null.synchronize()
sol = sol_gpu.get()
```

---

## Postprocessing

### Processors (2D and 3D)

```python
all_dofs = mlhp.DoubleVector(sol.tolist())
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.DoubleVector(indicator.ravel("C")), nvoxels=ncells, lengths=lengths)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([degree + 2] * D)
```

### 2D — DataAccumulator

```python
result = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, result, processors)

disp = np.array(result.data()[0])   # interleaved [ux0, uy0, ux1, uy1, ...], length = 2 * npoints
ux = disp[0::2]
ind_viz = np.array(result.data()[1])  # scalar per point, length = npoints

tri = result.triangulation()         # matplotlib Triangulation
tri.set_mask(ind_viz[tri.triangles].mean(axis=1) < 0.5)  # mask void triangles

fig, ax = plt.subplots()
cb = ax.tricontourf(tri, ux, cmap="turbo", levels=24)
fig.colorbar(cb)
ax.set_aspect("equal")
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.minorticks_off()
fig.tight_layout(pad=0)
```

`result.data()` index corresponds directly to the `processors` list order:
- `[0]` → displacement (vector, interleaved, length `D * npoints`)
- `[1]` → indicator (scalar, length `npoints`)

### 3D — PVtuOutput (ParaView VTU)

```python
out_stem = str(BASE_DIR / "output" / "elastic_3d")
Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
output = mlhp.PVtuOutput(filename=out_stem)
mlhp.basisOutput(basis, postmesh, output, processors)
print(f"VTU written to {out_stem}.pvtu")
```

---

## CT geometry files

The indicator geometry is generated once and saved as `.npz`. Drivers load it at runtime.

**Create** (run once per grid resolution):
```bash
python projects/16_elastic_fem/create_CT_2D.py --nx 80 --ny 40
python projects/16_elastic_fem/create_CT_3D.py --nx 80 --ny 40 --nz 20
```

Files land in `code/data/CT_2D.npz` and `code/data/CT_3D.npz`.

**Load in a driver**:
```python
ct_default = BASE_DIR.parent.parent / "data" / f"CT_{D}D.npz"
ct = np.load(args.ct or ct_default)
indicator = ct["indicator"]      # shape (Nx, Ny) or (Nx, Ny, Nz); values 1.0/1e-5
Lx = float(ct["Lx"])
Ly = float(ct["Ly"])

if D == 2:
    Nx, Ny = indicator.shape
    ncells = [Nx, Ny]
    lengths = [Lx, Ly]
    elem_lengths = [Lx / Nx, Ly / Ny]
else:
    Lz = float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    ncells = [Nx, Ny, Nz]
    lengths = [Lx, Ly, Lz]
    elem_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]
```

---

## Common pitfalls

| Mistake | Correct |
|---------|---------|
| `basis.ndofs()` | `basis.ndof()` (no trailing s) |
| `cp_splinalg.cg(..., rtol=...)` | `tol=` (cupyx doesn't accept `rtol`) |
| Geometric void mask `x²+(y−1)²<0.25` | Use `result.data()[1]` (indicator) for mask |
| Comparing K_ref from mlhp vs FEM.py directly | Different local node orderings; only the assembled global K needs to match |
| `K_ref` on a non-square element vs. FEM.py `s=2.0` | K_ref depends on element shape; use matching element dimensions |
| Uploading full matrix to GPU | Pass `K_ref.ravel("C")` — not the 2D array |
| `efts_gpu` as int64 | Must be `cp.int32` to match `const int*` in kernel |
