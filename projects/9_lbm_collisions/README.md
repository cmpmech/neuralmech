# 9_lbm_collisions

Collision operators of the lattice Boltzmann method for Chapter 9: a single relaxation rate
(BGK), two rates (TRT) and the cumulant collision on flows at high Reynolds numbers. Both
drivers need `cuvox` (lattice Boltzmann with the cumulant collision) and a GPU.

- `shear_layer2D.py` -- the thin doubly periodic shear layer of Minion and Brown at
  Re 1e5: BGK, TRT and cumulant on 256^2 nodes against the cumulant on 1024^2, vorticity
  at t = 0.55 and t = 1
- `cube3D.py -> data/lbm_cube.npz` -- flow past a cube at Re 1e4 in a cubic box of
  384^3 nodes (D3Q27): the steps after which BGK and TRT blow up, and the Q-criterion and
  streamwise vorticity of the cumulant run, rendered by `0_pvpython/render_lbm_cube.py`

## Non-obvious technicalities (authored by Claude)

BGK is run as the two-rate scheme with equal rates, i.e., the magic parameter
$\Lambda = (1/\omega^+ - 1/2)(1/\omega^- - 1/2)$ set to $9\nu^2$, and TRT at $\Lambda = 1/4$.
All runs are single precision. A run counts as blown up once a velocity exceeds ten times
the shear or inflow velocity, or a fluid node turns NaN; the 2D runs are checked 40 times,
BGK and TRT in 3D 400 times (every 38 steps), so the reported blow-up is the first check
after it. In the cube flow, BGK and TRT blow up at the upstream top edge of the cube, where the
flow separates (checked every 10 steps: BGK at step 364, TRT at step 115); the cumulant
run holds from the same impulsive start. The Q-criterion
$Q = -\tfrac{1}{2}\,\partial_j u_i\,\partial_i u_j$ is computed by central differences with
the solid nodes at rest, and the render drops it within six nodes of the cube, where the
staircase wall leaves spurious positive values.
