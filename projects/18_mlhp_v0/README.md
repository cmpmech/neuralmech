# Voxel Finite Elements and Topology Optimization

Immersed finite elements on voxel data, solved on the GPU, for **Chapter 18
(Simulation Acceleration via GPUs)**. Every driver builds its basis with `mlhp` and
integrates one preintegrated reference element over the voxel grid, so the whole
assembly reduces to scaling one small matrix per element.

## Drivers

- `topopt_elasticity2D_cg.py`
  compliance minimization of a half MBB beam with a matrix-free conjugate gradient
  solver on the CPU, the reference the GPU drivers are measured against
- `topopt_mlhp_cuda_subvoxel.py`
  the same optimization on the GPU, with several design voxels per finite element and
  a conjugate gradient solve
- `topopt_mlhp_cuda_subvoxel_direct.py`
  the same setup with a direct solve instead

Both GPU drivers take `--dim 2` or `--dim 3`.

## Voxelized elasticity

`voxelized/` holds the linear elastic solves the topology optimization drivers are
built on, in order of increasing sophistication.

- `create_CT_plate_hole_2D.py` -> `data/plate_hole_2D.npz`
- `create_CT_plate_hole_3D.py` -> `data/plate_hole_3D.npz`
  synthetic scan data, a plate with a hole
- `elastic.py`
  the plain immersed solve with standard quadrature
- `elastic_momentfitting.py`
  cut cells integrated by moment fitting instead
- `elastic_momentfitting_preintegrated.py`
  moment fitting with the reference element preintegrated once
- `elastic_matrixfree.py`
  the same solve without ever forming the global matrix
- `elastic_cuda.py`
  the matrix-free solve moved to the GPU
- `elastic_cuda_subvoxel.py`
  several voxels per element, so the material resolution outruns the mesh
- `elastic_cuda_subvoxel_multigrid.py`
  a geometric multigrid preconditioner over the same hierarchy
