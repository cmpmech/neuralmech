# Load-percolation fracture

Agent-developed research prototype for **Chapter 16** (agentic code
development): 3D brittle fracture of a cube with a spherical void under
uniaxial tension, computed without solving an elastic boundary-value problem --
distance transforms and curvature-anchored stress amplification, a GPU
random-walker transport probe, a max-flow/min-cut fracture surface, and an
analytic unzipping load-displacement curve. See `METHOD.md` for the paradigm
and `AGENT_LOG.md` for the full development narrative.

## Drivers

- `fracture_percolation.py`
  full pipeline: amplification field, walker probe, min-cut fracture surface,
  load-displacement curve _needs void_cube_64.npz_
- `calibrate_kt_2d.py`
  2D disk-in-square harness that verifies the SDF-curvature amplification
  against the exact Kirsch solution
- `verify_anchors.py`
  regression gate: every closed-form anchor (geometry, Kirsch, Goodier,
  min-cut, penny/Griffith) as pass/fail checks
- `validate_fem.py`
  CUDA voxel-FEM reference (adapted from `16_mlhp/voxelized/elastic_cuda.py`)
  and method-vs-FEM comparison; ground truth only, not part of the method
  _needs void_cube_64.npz and the saved fracture_percolation results_

## Data generation

- `create_void_cube.py` -> `data/void_cube_{32,64,128}.npz`

## Non-obvious technicalities (authored by Claude)

- **Curvature from voxel masks.** The level sets of a Euclidean distance
  transform are locally spheres around discrete boundary voxels, so their
  curvature is 1/(distance to that voxel), not the surface curvature. The
  pipeline therefore computes the shape operator of a Gaussian-smoothed
  indicator, tangentially smooths the *operator entries* before the
  eigendecomposition (averaging eigenvalues instead biases kappa1/kappa2 apart
  under staircase noise), and samples at the pulled-back surface point.
- **Net-section referencing.** The amplification is Kt * A_gross/A_net(z)
  (Peterson's finite-width convention). The walker probe confirms the plane
  factor independently: by flux conservation the mean midplane face flux is
  exactly A_gross/A_net times the far-field flux.
- **int32 flow quantization.** `scipy.sparse.csgraph.maximum_flow` stores flow
  in int32; capacities are scaled so the median face maps to 1e4 and clipped
  at 1e6, and the smallest z-plane capacity sum is asserted below 2^31.
- **Mesh-objective softening.** Breaking faces by strength alone would make
  softening resolution-dependent (near-tip stress at half a voxel grows like
  1/sqrt(h)); the unzipping instead takes the finite-fracture-mechanics
  envelope min(strength criterion, penny-crack toughness criterion), which is
  h-independent in the propagation regime.
- **Jump-accelerated walkers.** Away from the void and the z-walls, a walker's
  next k lattice steps cannot meet a boundary, so their endpoint is sampled in
  one shot from exact binomials, mirror-folded at the lateral walls. The
  per-iteration snapshot measure of such a jump chain is the time-stationary
  density divided by the local jump length -- walkers must be *initialized*
  from that measure, or every tally carries a slow transient bias (measured at
  -7% on the conductance before the fix).
