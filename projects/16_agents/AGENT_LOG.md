# Agent log -- load-percolation fracture

Narrative log of the agent sessions that produced this project: the conversation
with the author, the brainstorm (including rejected ideas), design decisions,
experiments, and dead ends. Newest entries at the bottom. Written by Claude
(Fable 5) as the work happened.

---

## Session 1 -- 2026-07-18

### The task (author's prompt, condensed)

> Tackle 3D fracture simulation in a completely different way than exists -- not
> using/reimplementing any method from the literature. It should be truly novel
> and potentially significantly faster. Benchmark: a cube with a sphere under
> uniaxial tension; result: load-displacement curve + fracture pattern. Must be
> extendable to CT-scan geometries later; random material fluctuations may
> emulate CT data during development. Keep logs of the conversation and the
> thought process in this directory.

### Round 1 brainstorm -- rejected as "not novel enough"

First proposals all kept a voxel FEM solve in the loop and layered a novel
fracture treatment on top:

- **A. Graph-cut fracture**: FEM stress field -> edge capacities -> crack as
  global min-cut via max-flow; F-u by energy-balance unzipping. (Best of round 1.)
- **B. Neural implicit crack surface**: crack = zero level set of a small neural
  field, optimized on the Francfort-Marigo energy with the repo's topopt adjoint.
- **C. Spectral fracture**: fracture load as the point where a strength-weighted
  graph Laplacian loses algebraic connectivity; crack = Fiedler vector nodal surface.
- **D. Woodbury/Green-function bond breaking**: FFT-diagonal pristine stiffness +
  low-rank updates per broken face.

Author's verdict: *"using FEM is not novel enough: try to find a completely new
paradigm."* Correct call -- A-D are novel algorithms but conventional paradigm
(solve the elastic BVP, then decide where the crack goes).

### Round 2 brainstorm -- FEM-free paradigms

Reframed the question as: can fracture be computed **without ever solving the
elastic boundary-value problem**? Elasticity may only enter through closed-form
local anchors or cheap stochastic probes. Every existing method family (FEM,
phase-field, XFEM, peridynamics, lattice/DEM, MPM, BEM) solves a global field
problem repeatedly; skipping it entirely is the actual paradigm claim.

- **P1. Geometric-combinatorial fracture** ("fracture as image processing +
  network flow"): distance transforms + morphology give load-bearing cross
  sections; local stress amplification from SDF curvature via *analytic* anchors
  (Kirsch, Goodier -- never fitted to FEM); strength capacities on voxel faces;
  **max-flow value = collapse load, min-cut = fracture surface**; F-u from
  analytic crack-compliance unzipping. O(N log N), deterministic, CT-native.
- **P2. Stochastic walker fracture** ("Monte Carlo fracture"): load carried by a
  GPU swarm of random walkers (injected at the loaded face, absorbed at the
  opposite one, reflected elsewhere); walker flux through voxel faces measures
  load transfer and concentrates at crack tips automatically; faces whose flux
  exceeds strength break and become reflecting -> avalanches and arrest emerge.
  Walk-on-spheres jumps (SDF-sized) make paths ~log N. Ancestry: the dielectric
  breakdown model of statistical physics (Niemeyer et al. 1984), but that was a
  2D fractal-pattern toy -- a quantitative 3D mechanics version with F-u output
  does not exist.
- **P3. Spin-Hamiltonian fracture**: crack = ground-state domain wall of an Ising
  system (interface energy = local strength, load tilts the Hamiltonian), solved
  by GPU annealing; temperature gives fracture-pattern ensembles and strength
  statistics for free. Deterministic limit coincides with P1's min-cut.

Novelty scans (web): random-fuse statistical physics knows the fracture <->
optimal-surface correspondence in the strong-disorder limit, and graph NNs are
used to *predict* crack paths, but no engineering solver computes fracture via
elasticity-free flow networks or walker swarms. Walk-on-spheres literature is
about solving PDEs pointwise, never about fracture. Clean enough to proceed.

### Decisions (author)

- Paradigm: **P1 + P2 hybrid** -- deterministic geometric/network-flow backbone,
  walker swarm as local stress probe where SDF curvature flags concentration.
- Benchmark geometry: **spherical void** (pore) at the cube center -- analytic
  stress concentration (Goodier ~2.045 at nu=0.3) at the equator, closest analog
  to CT porosity.

### Method name

**Load-percolation fracture**: load is a conserved flux routed through the
microstructure; fracture is the combinatorial failure of the routing network.

### Feasibility measurements (planning phase)

`scipy.sparse.csgraph.maximum_flow` (Dinic) on 6-neighborhood grid graphs with
supersource/sink, int64 CSR input (measured in this session):

| case | nodes | edges | time | RSS |
|---|---|---|---|---|
| 64^3 uniform capacities (+-10% noise) | 262k | 1.56M | 22.8 s | - |
| 64^3 with weak equatorial band (Kt-like) | 262k | 1.56M | 0.33 s | - |
| 96^3 weak band | 884k | 5.3M | 2.7 s | - |
| 128^3 weak band | 2.10M | 12.6M | 17.5 s | 0.93 GB |

Take-aways: the physically realistic case (weak band from stress concentration)
is the *fast* case for Dinic; the pathological near-tie uniform case only occurs
in the plain-cube anchor test, which is run at <= 64^3. scipy's internal flow
arrays are int32 -> capacities must be quantized so that total flow < 2^31
(median face capacity -> 1e4, cap at 1e6, a-priori overflow assert). PyMaxflow
is not installed; scipy suffices up to 128^3, so no new dependency.

Environment checks: cupy 14.1 works on GPU again (CUDA init was broken on
2026-07-14, fixed since), torch 2.9.1+cu128 sees the GPU,
`scipy.sparse.csgraph.maximum_flow` importable, networkx available (debug-only
fallback), PyMaxflow/igraph missing. No SDF / curvature / random-field /
marching-cubes utilities exist anywhere in the repo -- all new code.

### Validation doctrine

The method is calibrated **only** against closed-form anchors: Kirsch (2D hole,
Kt=3), Goodier (3D spherical void, Kt=(27-15nu)/(2(7-5nu))), plain-cube net
section, harmonic sphere concentration 3/2, Maxwell dilute conductance 1-3f/2,
dilute elastic stiffness deficit, penny-crack compliance (Tada). The existing
CUDA voxel FEM (`projects/16_mlhp/voxelized/elastic_cuda.py`) is used **only**
as reference ground truth in `validate_fem.py` -- never inside the method, never
for calibration. That separation is what keeps the "no BVP solve" claim honest.

### Known weaknesses recorded up front (to be quantified, not hidden)

- 6-neighborhood min-cut has an L1 metrication bias toward axis-aligned cuts;
  benign here (the true crack *is* axis-aligned) but it inflates apparent
  accuracy -- must be stated in METHOD.md; 18-neighborhood Cauchy-Crofton weights
  are the documented extension.
- The harmonic->elastic concentration mapping is exact at the anchors, degrades
  at saddles (crack fronts -> use the penny-crack K_I anchor there instead) and
  at re-entrant corners (absent in this benchmark, present in CT masks).
- Walker flux tallies next to stair-step boundaries are biased; read
  concentrations from shells >= 1-2 voxels away and extrapolate with the known
  analytic decay shape.

### Plan of record

M0 geometry+scaffolding -> M1 2D Kirsch calibration -> M2 3D geometric Kt ->
M3 GPU walkers -> M4 min-cut -> M5 unzipping F-u -> M6 FEM comparison ->
M7 random-strength ensemble + docs. Files: `helper.py` (method library),
`create_void_cube.py` (data gen), `calibrate_kt_2d.py`, `fracture_percolation.py`
(main driver), `verify_anchors.py` (regression gate), `validate_fem.py`
(reference comparison), `README.md`, `METHOD.md`, this log.

---

## Session 1 -- implementation notes

### M0 (geometry) -- pass

`create_void_cube.py` writes 32/64/128 voxel cubes (npz convention shared with
`16_mlhp/voxelized`). Solid fraction matches 1 - 4/3 pi R^3 to 3e-4 at 128^3;
SDF error < h/2 near the void at every resolution; net-section counting
converges to the exact mid-plane area.

### M1 (2D Kirsch harness) -- pass, after two dead ends

Dead end 1: computing level-set curvatures from the EDT distance field and
pulling them back analytically with kappa0 = kappa/(1 - kappa d). Fails badly:
an EDT field's level sets are locally spheres around the *discrete boundary
voxel centers* (Voronoi cells), so their curvature is 1/(distance to that
voxel), not the surface curvature. Measured curvature errors up to 180%.

Dead end 2: curvature from a Gaussian-smoothed indicator (sigma 1.5 voxels),
sampled at the pulled-back surface point x - d n. Mean is right but staircase
ripples survive the smoothing: pointwise std ~ 100% of the true curvature.

Fix that works: indicator smoothing sigma = 2 voxels + *band-weighted
tangential smoothing of the curvature field itself* (weights = |grad chi|,
sigma = 4 voxels) before the pullback sampling. Surface curvature varies slowly
along a surface, so this denoises without bias: circle curvature mean 9.90,
std 0.85 (exact 10) at R = 25.6 voxels.

Amplification model (all-analytic, no fitted constants): both anchor ligament
profiles collapse onto Kt = 1 + (Kt_surf - 1) [w q^(2+s) + (1-w) q^(4+s)] with
q = 1/(1 + kappa1 d), shape s = kappa2/kappa1 (0 cylinder, 1 sphere), Kt_surf
angle-blended between equator and pole anchor values (exactly 3 - 4cos^2 theta
for Kirsch). Result at 256^2, R = 0.1: max pointwise error vs the exact Kirsch
ligament profile **3.2%**, curvature max err 13% (noise, enters only the decay
length). The "calibration" harness calibrates nothing -- it verifies the
discrete pipeline against closed forms.

### M2 (3D geometric Kt) -- pass after one more fix

At 128^3 the two principal curvatures split apart (5.55/4.42 instead of 5/5):
noise-induced eigenvalue repulsion, which biases the shape parameter s toward
"cylinder" and overshoots Kt (2.27 vs 2.045). Fix: tangentially smooth the
*shape-operator entries* before the eigendecomposition (eig of mean, not mean
of eigs). After: kappa1 ~ kappa2, ligament profile error <= 3.4% vs exact
Goodier at 32/64/128, surface Kt -> 2.000 at 128^3 (voxel centers sit half a
cell inside the surface, so a slight undershoot is expected).

### M3 (walker probe) -- pass

Design: pure-lattice GPU walkers (torch), blocked-move = stay reflecting BC,
absorb below bottom + respawn on top (constant population -> steady state),
init from the linear-in-z profile to cut burn-in, integer tallies so GPU
atomics stay deterministic. Anchors at 64^3 with 2M walkers x 6144 steps:
plain-cube conductance 0.9994 (exact 1), void conductance ratio 0.9478 vs
Maxwell dilute 0.9497, flux-excess profile matches the harmonic ligament to
<1% once the finite-cube confinement baseline is divided out. The raw rings
sit ~4-5% above the *dilute* formula because the insulating lateral walls
really do squeeze flux inward -- the probe measures a real finite-size effect,
which later turned out to be the key to the FEM comparison. Runtime ~40 s
(kernel-launch bound; walk-on-spheres jump acceleration is the documented
upgrade, not needed for v0).

### M4 (min-cut) -- pass

Plain cube: flow value exactly strength*L^2 (quantization-exact), cut = one
flat plane. Void cube: the cut is exactly the midplane through the void (all
3572 faces at z = 31|32), load bound 0.732 gross, 0.25 s at 64^3. Capacity
convention: c = strength * A_face / amp, so the flow value is a
fully-redistributed rupture bound; initiation = min strength/amp; the true
brittle peak comes from the unzipping between those brackets.

### M5 (unzipping) -- pass

Mesh-objectivity decision: breaking faces by strength alone makes softening
h-dependent (near-tip stress at half a voxel ~ 1/sqrt(h)); instead each face
breaks at the finite-fracture-mechanics envelope min(strength criterion,
penny-crack toughness criterion K_I = 2 sigma sqrt(c/pi) = K_Ic), which is
h-independent in the propagation regime. Penny compliance checked against
Griffith (dU/dc = G 2 pi c) to machine precision. Resulting curve: linear ramp,
initiation at the void ring, short stable growth, peak, snap-back softening
(u retreats while F drops - physical for a large crack in a small specimen),
terminal drop appended at full separation (the infinite-body penny compliance
cannot reach K=0 by itself - documented limitation).

### M6 (FEM reference) -- the one real surprise

First comparison: FEM equator Kt (shell-averaged + extrapolated, because the
voxel staircase makes a raw stress max meaningless - it spikes to 3.7) came
out 2.32, i.e. 13% above Goodier. Cause: finite specimen (net section 0.874 +
wall confinement) - the same ~5% enhancement the walkers had already measured
in the harmonic problem. Fix inside the paradigm: reference the concentration
to the net section (Peterson's finite-width convention, amp = Kt * A_g/A_net),
which the walker plane-flux conservation confirms independently. After: 

| quantity | method | FEM | error |
|---|---|---|---|
| K0 | 0.9306 | 0.9328 | -0.2% |
| surface amplification | 2.345 | 2.316 | +1.2% |
| initiation load | 0.460 | 0.432 | +6.6% (partly bookkeeping: face-average vs surface extrapolation) |

FEM solve: 4.5 s for ONE linear solve at 64^3 (float64 diagonal-PCG CUDA).
The method's deterministic backbone runs in ~1 s total.

### M7 (randomness + ensemble)

Lognormal strength field (unit median), corr 4 voxels. sigma=0.5: initiation
0.234 (weakest link), peak 0.493, bound 0.779 (redistribution profits from
strong regions); cut relocates to a weaker parallel plane (z=0.516) but stays
planar - the LATERAL=0.3 drive prices staircase risers ~4x a plane face.
sigma=1.0 with LATERAL=0.6: genuinely rough cut (z in [0.50, 0.59], 3937 faces
vs 3572 planar), scattered weak-patch initiation, pre-peak nonlinearity -
emergent quasi-brittle behavior. Disorder monotonically lowers knee and peak.

### 128^3 stretch run (deterministic)

K0 0.9311 (64^3: 0.9306 - converged), initiation 0.4373 (64^3: 0.4601,
FEM-derived target 0.432 - converging with h as the face-averaged
amplification approaches the surface value), peak 0.5527 (64^3: 0.5474, ~1%),
cut exactly planar (14328 faces = solid midplane). Timings: geometry 5.4 s,
walkers 219 s (unoptimized lattice walk - the acknowledged hotspot), min-cut
17.9 s (matches the planning benchmark), unzip < 0.1 s.

### Where this leaves the project

Working v0 of a genuinely FEM-free 3D fracture pipeline with quantified
accuracy (stiffness -0.2%, amplification +1.2%, initiation +7% at 64^3
improving to +1% at 128^3 against a FEM reference it was never calibrated on)
and a deterministic backbone about three orders of magnitude faster than a
conventional fracture simulation. Obvious next steps, in value order:
1. walk-on-spheres jump acceleration for the walkers (40 s -> ~1 s at 64^3);
2. 18-neighborhood Cauchy-Crofton capacities (kill the L1 metrication bias,
   needed before claiming crack *paths* in anisotropic stress states);
3. walker-measured compliance of the cracked domain (replace the infinite-body
   penny formula near full separation);
4. crack-front amplification from walker tallies around the growing cut
   (adaptive re-ranking during unzipping instead of static overload order);
5. a real CT mask end-to-end.

---

## Session 2 -- 2026-07-18

### The task

Continue with next step 1: jump acceleration for the walker probe, the
runtime hotspot (42 s of the 43 s pipeline at 64^3; 219 s at 128^3).

### Design: exact jumps, not walk-on-spheres

Genuine walk-on-spheres would change the discretization (continuum Brownian
motion, boundary projections); instead the lattice chain is kept *exactly* and
only sampled faster. A walker whose safe radius
k = min(floor(EDT to void) - shell, z, res-1-z, kmax) is >= 2 cannot meet the
void, the absorbing bottom, or the tallied top layer within k steps (L1
displacement <= k), so the endpoint of k constrained steps equals a free
k-step walk: sampled in one iteration as three exact binomials (axis counts
via B(k, 1/3), B(k-n1, 1/2); +- splits via B(n, 1/2); `torch.binomial`
verified deterministic-under-generator and distribution-exact against
brute-force k-step ensembles, max histogram deviation at noise level).
Two structural tricks:

- **Lateral walls don't bound jumps.** Blocked-move-stays reflection is
  exactly the half-integer mirror fold of a free walk, so jump endpoints are
  folded back with period 2*res. The z-walls must still bound k: the bottom
  absorbs, and the top layer's occupancy tally is a time average that stays
  exact only if the top is reachable solely at a jump's *final* step
  (k <= res-1-z guarantees exactly that).
- **Tallies by endpoint identity.** Net crossings of any z-plane by any path
  equal the endpoint side difference, so per-plane flux is tallied exactly for
  all walkers by two bincounts on a difference array (respawn teleports
  excluded by tallying pre-respawn). Per-face tallies remain exact inside the
  shell -- jump paths provably stay >= shell voxels from the void -- which is
  the only region where per-face flux is read (`amp_walker` is zeroed
  elsewhere; the far-field normalization now uses the exact plane aggregates,
  where flux conservation makes every far plane carry the full current).

### The trap: snapshot measure of a jump chain (half a session's debugging)

First full run: conductance 0.8863 instead of ~0.948, with the jump kernel
provably exact. Bisection: single-step runs fine, jump runs biased, bias
*not* monotone in jump size -- so not the sampler. Cause: the per-iteration
snapshot measure of a jump chain is not the time-stationary density n(x) but
n(x)/k(x) (slow cells collect snapshots). Initializing walkers from the
time-stationary profile (z+1) therefore starts the *iteration* chain far from
its stationary measure, and the resulting redistribution transient decays too
slowly to burn away -- it biased every tally. Fix: initialize from
(z+1)/k_eff(x). After the fix, jump runs converge onto the long single-step
control (0.9493/0.9494 vs 0.9499) with burn-ins of a few hundred iterations.
Corollary discovered on the way: v0's numbers were themselves slightly
under-burned -- the converged 64^3 conductance is 0.9496 (vs 0.9478 logged;
sits almost exactly on Maxwell's 0.9497), and the converged 128^3 value is
0.9520 (the logged 128^3 K0 0.9311 came from a 2048-step burn-in where ~9000
walker-steps are needed; converged K0 is 0.9359). k0 vs FEM improves from
-0.2% to -0.1%.

### Estimator cleanup

Absorption rate is now Rao-Blackwellized: a bottom-layer walker absorbs with
probability exactly 1/6, so occupancy/6 replaces realized absorption counts
(shot noise removed, integers kept, dynamics unchanged). Remaining conductance
scatter is density-fluctuation limited: +-0.2% at the production budget
(2M walkers x ~4100 measured steps), which bounds K0 at +-0.3% -- the same
statistical class as v0, now bias-free.

### Results

- 64^3 production: walkers 29 s (was 42 s), pipeline unchanged elsewhere;
  k0 0.9323 vs FEM 0.9328 (-0.1%); deterministic backbone bit-identical
  (initiation 0.4601, peak 0.5474, same 3572-face planar cut).
- 128^3: walkers 71 s converged (was 219 s *and* under-burned): ~3x faster
  and correct. steps/iteration 2.7 at 64^3, 5.6 at 128^3 (the harmonic-mean
  law: iteration gain = 1/E[1/k], so the single-step shell dominates; the
  gain grows with resolution as the shell's volume fraction shrinks).
- verify_anchors.py now also gates the walker chain: plain-cube conductance
  vs exact 1 and void conductance vs Maxwell dilute.
- validate_fem comparison rerun: k0 -0.1%, amplification +1.7% (the +1.2% in
  the M6 table was stale relative to the final v0 code state -- current
  deterministic amp.max() is 2.356, unchanged by this session), initiation
  +6.6% unchanged.

### Honest assessment of the speedup

The session-1 estimate "40 s -> ~1 s" was optimistic: it ignored that the
stationary snapshot measure concentrates walkers in the single-step cells
(1/E[1/k], not E[k]) and the cost of exact binomial sampling (~30% of the
iteration). Measured: 1.45x at 64^3, ~3x at 128^3, growing with resolution --
plus the bias fixes, which matter more than the wall clock. The remaining
levers, in value order: variance reduction on the density fluctuations (the
budget is statistics-bound, not step-bound; common-random-number plain/void
ratio estimators are the candidate), a fused/compiled iteration kernel, and a
two-phase run (small-shell conductance pass + short per-face pass).
