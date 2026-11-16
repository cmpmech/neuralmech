# Dynamic Mode Decomposition

Exact dynamic mode decomposition of the cylinder wake for **Chapter 10 (Machine
Learning in Computational Mechanics)**, on the lattice Boltzmann flow of
`8_physics_drivers/fluid2D.py` at reduced resolution.

## Data generation

- `cylinder_snapshots.py` -> `data/cylinder_snapshots.npz`
  vorticity snapshots behind the cylinder from the impulsive start to 120 convective
  times, on the GPU (cufluid)

## Drivers

- `dmd_cylinder.py` _needs `cylinder_snapshots.npz`_
  DMD trained on the periodic shedding and on the transient start-up: eigenvalues,
  leading modes, and the forecast error beyond the training window

## Non-obvious technicalities (authored by Claude)

Time is in convective units $D/U$, so the DMD frequencies are Strouhal numbers
$\mathrm{St}=fD/U$. The shedding at $\mathrm{St}=0.27$ is faster than the $0.16$ to $0.2$ of an
unbounded cylinder at $\mathrm{Re}=100$: the channel is only four diameters high (25 %
blockage), and the coarse lattice (24 nodes across the cylinder) adds to it.

Vorticity is stored in units of $U/D$, zero inside the cylinder, and the images use a
symmetric color range at the 99th percentile of the final snapshot, because the thin
boundary layer on the cylinder wall peaks far above the wake. The forecast
$\mathbf{x}_k=\boldsymbol{\Phi}\boldsymbol{\Lambda}^k\mathbf{b}$ is evaluated from the first snapshot of the
training window; for earlier snapshots ($k<0$) it diverges and is not exported.

The start-up window (2.5 to 22.5 $D/U$) contains the growth of the instability, which no
fixed set of eigenvalues on or near the unit circle describes: DMD returns eigenvalues
off the circle (radius up to 1.03) at wrong frequencies, misfits its own training window
by 38 %, and its forecast grows without bound.
