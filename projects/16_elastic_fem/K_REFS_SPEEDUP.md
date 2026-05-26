# K_refs speedup — notes for C++ extension

## The problem

`elastic_mlhp_cuda_subvoxel.py` pre-computes `K_refs[s]` — the per-subvoxel reference
stiffness matrices — by calling `mlhp.integrateOnDomain` once per subvoxel `s`.
For `DEGREE=3, SUB_VOXELS=8, D=3`: `n_sub = 8³ = 512` calls, each on a 1-element macro
mesh with `nsubcells=[8,8,8]` (512 subcells internally).  Each call wastes 511/512 of its
integration effort because only one subvoxel has a non-zero integrand.

Current wall time: **~100–300 s** for the first run (then cached to `.K_refs_<hash>.npy`).

---

## What the mlhp source says

Source lives at `solvers/mlhp/` (pinned at 0.1.2).

### No per-element output exists

`src/core/assembly.cpp` (around line 189–251):

```
for each element:
    for each quadrature point:
        basis.evaluateSinglePoint() → BasisFunctionEvaluation
        integrand.evaluate()       → accumulates into localTargets (dense element matrix)
    assemble()                     → scatter localTargets into global sparse matrix
    // localTargets discarded here — no way to intercept from Python
```

There is no Python hook to capture the local element matrix before it is scattered.

### What IS exposed to Python

- `mlhp.gridQuadrature(nsubcells=[S]*D).evaluate(mesh, cell, orders)` returns `rst`, `xyz`,
  `weights` for all quadrature points.  Could be used for a NumPy reimplementation (but you
  would also need shape function derivatives, which are **not** exposed).
- `BasisFunctionEvaluation<D>` (basisevaluation.hpp) has no pybind11 bindings at all.
- `mlhp.domainIntegrand(callback, types, diffOrder)` exposes a Python-callback integrand, but
  the callback still writes into a global sparse target — local matrices are not accessible.

---

## Planned fix: small C++ pybind11 extension

Write a standalone pybind11 module (call it `mlhp_ext.so` or similar, compiled via a tiny
`CMakeLists.txt` or `setup.py`) that:

1. Calls mlhp's C++ headers directly (mesh, basis, quadrature, material evaluation).
2. Runs a **single integration pass** over the one-element reference mesh with
   `nsubcells=[S]*D`.
3. For each quadrature point, looks up which subvoxel it belongs to (integer division of its
   reference coordinate) and accumulates the local `B^T C B * w * |J|` contribution into
   `K_refs[s]` instead of a global sparse matrix.
4. Returns `K_refs` as a `(n_sub, ndof_e, ndof_e)` numpy array.

Expected speed: one C++ integration loop, **~0.1–0.5 s** for p=3 3D.

### Key mlhp C++ types to use

| Purpose | Header / Class |
|---|---|
| Mesh | `mlhp/core/mesh.hpp` — `RefinedGrid<D>` |
| Basis | `mlhp/core/basis.hpp` — `HpTrunkSpace<D>` |
| Quadrature | `mlhp/core/quadrature.hpp` — `GridQuadrature<D>` |
| Basis evaluation | `mlhp/core/basisevaluation.hpp` — `BasisFunctionEvaluation<D>` |
| Material (isotropic elastic) | `mlhp/core/materials.hpp` — `IsotropicElasticMaterial<D>` |
| Plane stress | `mlhp/core/materials.hpp` — `PlaneStressMaterial` |

The integration loop to replicate is in `src/core/assembly.cpp`.  Copy/adapt the inner loop:
evaluate basis, evaluate constitutive tensor C, form B, accumulate `B^T C B * w` into the
right `K_refs[s]` slot.

### Subvoxel index from reference coordinates

For a reference element mapped to `[0, Lx] x [0, Ly] x [0, Lz]` with S subcells per
direction:

```cpp
int sx = (int)(rst[0] * S);  // rst in [0,1] (check mlhp's reference element convention)
int sy = (int)(rst[1] * S);
int sz = (int)(rst[2] * S);  // sz=0 for 2D
int s  = sx * (S * Sz) + sy * Sz + sz;  // C-order, Sz=S (3D) or 1 (2D)
```

This must match the Python-side indexing in `elastic_mlhp_cuda_subvoxel.py` and in
`cuda_assemble_K_e` in `mlhp_kernels.cu`.

### Files to create

```
projects/16_elastic_fem/
    mlhp_kref_ext/
        CMakeLists.txt      # finds mlhp headers + pybind11
        kref_ext.cpp        # the pybind11 module (~150-200 lines)
    build_kref_ext.sh       # one-liner: cmake + make, drops mlhp_kref_ext.so here
```

### Integration into the driver

Replace the `else` branch in `elastic_mlhp_cuda_subvoxel.py`:

```python
import mlhp_kref_ext  # built .so in the same directory

K_refs = mlhp_kref_ext.compute_k_refs(
    D, DEGREE, SUB_VOXELS, macro_lengths, NU
)  # returns np.ndarray (n_sub, ndof_e, ndof_e), ~0.1-0.5s
np.save(cache_file, K_refs)
```

The multiprocessing fallback (`_k_refs_worker` / `mp.Pool`) can stay as a backup for
environments where the extension is not built.

---

## Validation

After building:

```python
# Sum over subvoxels must equal the full E=1 integration
K_total_ext = K_refs.sum(axis=0)

# Reference: single mlhp call with E=1 everywhere
E_field_one = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector([1.0] * n_sub), nvoxels=[SUB_VOXELS]*D, lengths=macro_lengths)
c_one = mlhp.isotropicElasticMaterial(E_field_one, nu_field)
...
K_total_mlhp = np.array(m_ref.todense())

np.testing.assert_allclose(K_total_ext, K_total_mlhp, rtol=1e-10)
```

Also run the full solve and compare `max displacement` to the multiprocessing-computed K_refs.

---

## Related files

| File | Role |
|---|---|
| `elastic_mlhp_cuda_subvoxel.py` | Driver — contains `_k_refs_worker` (current slow path) |
| `mlhp_kernels.cu` — `cuda_assemble_K_e` | CUDA kernel that consumes K_refs on GPU |
| `solvers/mlhp/src/core/assembly.cpp` | mlhp assembly loop to replicate in C++ |
| `solvers/mlhp/src/core/basisevaluation.hpp` | Shape function evaluation types |
| `solvers/mlhp/src/core/materials.hpp` | Constitutive tensor types |
