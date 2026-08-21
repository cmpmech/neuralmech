# Load-percolation fracture -- method memo

3D brittle fracture computed **without ever solving an elastic boundary-value
problem**. Every existing fracture-simulation family (FEM + cohesive/XFEM,
phase-field, peridynamics, lattice/DEM, MPM, BEM) discretizes and repeatedly
solves a global field problem; here the crack surface, the peak load, and the
load-displacement curve come from image processing, combinatorial network flow,
and a stochastic particle probe. Elasticity enters only through closed-form
anchor solutions and through measured random-walk statistics.

Paradigm sentence: *load is a conserved flux routed through the microstructure;
fracture is the combinatorial failure of the routing network.*

## Pipeline

1. **Geometry (image processing).** Voxel mask (CT-native). Signed distance
   field by Euclidean distance transform; load-bearing net section per slice by
   voxel counting.
2. **Geometric amplification (P1).** Local stress concentration Kt(x) from the
   principal curvatures of the nearest void surface, with an analytic model
   that is exact for both closed-form anchors: surface value blended between
   the cylinder (Kirsch, Kt=3) and sphere (Goodier, Kt=(27-15nu)/(2(7-5nu)))
   cases by the curvature shape ratio s = kappa2/kappa1, angular dependence
   exact at equator/pole, ligament decay w q^(2+s) + (1-w) q^(4+s) with
   q = 1/(1+kappa1 d). Amplification is referenced to the net section
   (Peterson's finite-width convention): amp = Kt * A_gross/A_net(z).
3. **Walker probe (P2).** GPU swarm of lattice random walkers: injected at the
   loaded face, absorbed at the opposite face, reflected elsewhere; their
   steady-state face-crossing statistics measure the harmonic (conductivity)
   load-flux field, which concentrates at geometric features and crack fronts
   like a real stress field. Measured conductance -> initial stiffness K0 via
   the dilute harmonic-to-elastic anchor ratio; measured flux concentrations
   cross-check the geometric Kt where curvature is under-resolved. Walkers
   more than a shell away from the void take SDF-sized multi-step jumps that
   are exact in distribution (binomial endpoint sampling, mirror-folded at the
   reflecting walls), so wall clock scales with the jump-chain iteration count
   rather than the lattice step count.
4. **Fracture surface (min-cut).** Graph over solid voxels (6-neighborhood);
   face capacity = strength * area / amp (lateral faces carry reduced driving).
   The max-flow value between the loaded faces is a fully-redistributed rupture
   bound; the min-cut is the fracture surface. One combinatorial solve,
   0.3 s at 64^3, 18 s at 128^3.
5. **Load-displacement (unzipping).** Cut faces break in order of static
   overload; the load at each break event is the finite-fracture-mechanics
   envelope min(strength criterion via amp, toughness criterion via the exact
   penny-crack K_I = 2 sigma sqrt(c/pi)); stiffness evolves by the analytic
   penny compliance (Griffith-consistent to machine precision). Gives
   initiation, stable growth, peak, softening, and snap-back under
   displacement control.

## Calibration doctrine

No constant is fitted to a numerical reference. The only inputs are E, nu,
strength, toughness, and closed-form anchors: Kirsch, Goodier, plain-cube net
section, harmonic sphere concentration 3/2, Maxwell dilute conductance,
dilute elastic stiffness deficit, penny-crack compliance (Tada). The CUDA voxel
FEM in `validate_fem.py` is reference ground truth only.

## Validation snapshot (64^3, R=0.2 void, nu=0.3)

| quantity | method | reference | error |
|---|---|---|---|
| Kirsch ligament profile (2D, 256^2) | pipeline | exact | max 3.5% |
| Goodier ligament profile (3D) | pipeline | exact | max 3.4% |
| plain-cube conductance (walkers) | 0.997 | 1 (exact) | 0.3% (seed scatter ~0.4%) |
| void conductance ratio | 0.9493 | 0.9497 (Maxwell dilute) | 0.04% |
| initial stiffness K0 | 0.9323 | 0.9328 (FEM) | -0.1% |
| surface amplification | 2.356 | 2.316 (FEM, extrapolated) | +1.7% |
| initiation load | 0.460 | 0.432 (FEM + strength criterion) | +6.6% |
| plain-cube min-cut load | 1.0 | 1 (exact) | quantization-exact |
| fracture plane | flat equatorial cut through the void | expected | - |

Timings (64^3): geometry 0.7 s, walkers 29 s (jump-accelerated; 42 s before),
min-cut 0.3 s, unzip < 0.01 s. At 128^3: geometry 5.4 s, walkers 71 s
converged (the pre-acceleration 219 s run was also under-burned-in), min-cut
17.9 s; peak load moves by ~1% and initiation converges toward the
FEM-derived value (0.437 vs 0.432).
The FEM reference needs 4.5 s for *one* linear solve; a conventional fracture
simulation needs hundreds to thousands of such solves (phase-field also needs
a mesh fine enough to resolve its regularization length), so the deterministic
backbone (P1, ~1 s) is roughly three to four orders of magnitude faster than a
conventional run, and the full hybrid including the walker probe roughly one
to two.

## Honest limitations (quantified or flagged, not hidden)

- **Metrication bias.** The 6-neighborhood cut functional measures L1-projected
  area, overweighting oblique surfaces by up to sqrt(3). Benign here (the true
  crack is axis-aligned) which *flatters* the benchmark; 18-neighborhood
  Cauchy-Crofton capacities are the documented extension.
- **Harmonic-to-elastic mapping.** Exact at the anchors (sphere, cylinder);
  degrades at saddles (crack fronts use the penny K_I anchor instead) and at
  re-entrant corners (absent in this benchmark, present in CT masks).
- **Penny-compliance model.** Infinite-body formula: cannot reach zero
  stiffness at full separation (the terminal drop to F=0 is appended, not
  emergent), and underestimates compliance once the crack approaches the
  lateral faces.
- **Local curvature model.** Captures blob-like features; elongated ellipsoids
  need more than the two local curvatures. Walker flux corrects this in
  principle; the blending is v0.
- **Initiation bookkeeping.** The method's initiation uses face-averaged
  amplification half a voxel inside the surface; the FEM comparison
  extrapolates to the surface. Part of the +6.6% gap is this convention.
- **Walker per-face tallies.** With jump acceleration, per-face flux is exact
  only within the shell around the void (jump paths provably never enter it);
  outside it only exact per-plane aggregates exist, and `amp_walker` is zeroed
  there. Conductance/K0 precision is density-fluctuation limited (~0.3% seed
  scatter at the default budget); absorption shot noise is already removed by
  a Rao-Blackwellized bottom-occupancy tally.

## CT extension and material randomness

Geometry enters exclusively as a voxel mask in the same npz convention as
`projects/16_mlhp/voxelized` (CT scans drop in unchanged). Material
fluctuations enter as lognormal random fields on strength (and optionally
toughness) via `helper.lognormal_field` -- set STRENGTH_SIGMA > 0 in the driver.

## Relation to existing ideas

Statistical physics proved fracture <-> minimal-cut correspondence for fuse
networks in the infinite-disorder limit (Roux/Hansen and successors); the
dielectric breakdown model grows fractal discharge patterns with harmonic
measure; rigid-plastic limit analysis bounds collapse loads variationally;
Peterson's handbook references concentrations to net sections. None of these
is a quantitative 3D fracture *simulator*; the combination -- anchored geometric
amplification + walker transport measurements + capacity-graph min-cut +
finite-fracture-mechanics unzipping producing F-u curves -- appears to be new.
