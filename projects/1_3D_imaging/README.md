# 3D Imaging

Imaging inverse problems for **Chapter 1 (Computational Mechanics Meets Artificial
Intelligence)**: recover an image from incomplete or indirect measurements.

## Data generation

- `graded_fiber_gen.py` -> `data/graded_fibers_128.npy` (train),
  `data/graded_fibers_test_128.npy` + `data/graded_fiber_masks_test_128.npy` (test)
  random non-overlapping graded circles ("fibers") and the random pixel masks used
  as measurements

## Drivers

- `ct.py`
  computed tomography: Radon transform (sinogram) and filtered backprojection on a
  smooth phantom
- `mri.py`
  MRI compressed sensing: undersampled k-space and an iterative total-variation
  reconstruction of the Shepp-Logan phantom
- `ultrasound.py`
  pulse-echo ultrasound from a single emitter: Born forward model and
  delay-and-sum reconstruction of a point defect
- `ultrasound_fullsource.py`
  the same ultrasound setup with every transducer emitting in turn, summed over
  emitters
- `image_reconstruction_tv.py` _needs the graded-fiber test data_
  classical inpainting of the masked fibers: zero-fill, or total variation solved
  with ADMM
- `image_reconstruction_modl.py` _needs the graded-fiber test data and `models/modl_128.pt2`_
  the same inpainting with a learned MoDL network
- `train_modl.py` _needs the graded-fiber train data_
  trains the MoDL network (UNet denoiser + unrolled conjugate-gradient data term)
- `helper.py`
  shared operators: masking forward/adjoint, conjugate gradient, the `MoDL`
  module, and the ultrasound Born/delay-and-sum routines

## Non-obvious technicalities (authored by Claude)

### Total-variation inpainting via ADMM (`image_reconstruction_tv.py`)

The reconstruction solves

$$\min_x \tfrac{1}{2}\lVert Ax-b\rVert_2^2 + \lambda\lVert Dx\rVert_1,$$

where $A$ keeps the measured pixels (the mask) and $D$ is the discrete image
gradient. The $\ell_1$ term is non-smooth, so ADMM splits it off with an auxiliary
variable $z = Dx$ and a dual variable $u$, then alternates three cheap steps:

- **x-update** (smooth quadratic): solve $(A^\top A + \rho D^\top D)\,x = A^\top b + \rho D^\top (z-u)$
  with conjugate gradient. $D^\top D$ is the periodic-BC Laplacian (`lap`).
- **z-update** (the prox of the $\ell_1$ term): soft-threshold $Dx + u$ at
  $\lambda/\rho$.
- **dual update**: $u \leftarrow u + Dx - z$.

`RHO` is the ADMM penalty; `LAMBDA` the TV weight. The split is what lets a hard
non-smooth problem reduce to a linear solve plus a pixel-wise shrinkage.

### Born forward model and delay-and-sum (`helper.py`, `ultrasound.py`)

`born_forward` builds the A-scans (sensor-vs-time traces) under single-scattering:
each grid cell with reflectivity $z>0$ re-emits the incident pulse, arriving at
sensor $j$ after the round-trip time of flight

$$\tau = \big(\lVert p - p_\text{emit}\rVert + \lVert p - p_j\rVert\big)/c,$$

so the trace is $\sum_\text{cells} z\,\text{pulse}(t-\tau)$ plus the direct
emitter-to-sensor arrival. `das_backproject` inverts this by delay-and-sum: each
pixel sums, over all sensors, the recorded amplitude at its own time of flight;
the defect is where contributions add coherently.

- A single emitter only constrains the round-trip distance, so each sensor maps a
  reflection to an **ellipse** of possible positions — the reconstruction shows
  arcs and localizes weakly. This is expected, not a bug: the peak still lands on
  the defect, but the background is smeared.
- `ultrasound_fullsource.py` fires every emitter and sums the backprojections, so
  the ellipses intersect at the defect and the arcs suppress — a much sharper image.
