# Latent-space topology optimization

Topology optimization performed in the latent space of a generative model,
supporting **Chapter 15 (Generative Artificial Intelligence)**. The design is
decoded from a latent code rather than parameterized voxel-by-voxel, so every
candidate is constrained to lie on the learned manifold of fiber microstructures.

## Drivers

- `topopt_latent_ae.py` _needs `models/fiber_ae_1_256.pt2`_
  compliance minimization of a half MBB beam whose density field is the output of the
  fiber autoencoder. Adam optimizes the latent code, the volume constraint is an
  augmented Lagrangian, and the decoded fibers are read as holes so the surrounding
  matrix stays connected and load-bearing
- `topopt_latent_vae.py` _needs `models/fiber_vae_256_4.0_256.pt2`_
  the same optimization in the latent space of the fiber variational autoencoder. The
  design variable is the code of the second stage, whose prior is a standard normal,
  and a penalty on its negative log density keeps the design typical of the training
  set
- `topopt_latent_diffusion.py` _needs `models/fiber_diffusion_200_256.pt2`_
  the same optimization inside the fiber diffusion model. The design variable is the
  noise the sampler starts from, the deterministic reverse chain is differentiated end
  to end, and the noise is constrained to the shell of the standard normal instead of
  being penalized towards its mode
- `topopt_latent_reference.py`
  the classical density-based optimization the latent drivers are compared against:
  the same beam, discretization and volume fraction, but every voxel is a design
  variable of its own, updated by the optimality criterion. It reaches a compliance of
  30.5 against 30.8 for the autoencoder, 32.4 for the variational autoencoder and 31.6
  for the diffusion model, all at a volume fraction of 0.6

The three generative models are trained in `../15_anomaly/` (`fiber_ae_train.py`,
`fiber_vae_train.py`, `fiber_diffusion_train.py`).

## Non-obvious technicalities (authored by Claude)

The autoencoder only ever emits arrangements of circular fibers, which cover roughly
4-28% of the domain. Read as material, those fibers would be disconnected blobs with no
load path, so the density is inverted (`rho = 1 - decoded`): the fibers become voids
punched out of a solid plate. A decoded training sample starts around 0.89 material
fraction; reaching the target of 0.6 forces the optimizer off the strict fiber
manifold, where the holes grow and merge into elongated voids.

The compliance sensitivity is computed on the design grid exactly as in a classical
solver, then handed to autograd as the incoming gradient of `rho`
(`rho.backward(sensitivity)`); backpropagation through the decoder turns it into a
gradient on the latent code. The design first migrates to the target fraction and then
rearranges its holes toward the load path at (nearly) fixed volume.

None of the three latent drivers penalizes the intermediate densities, `PENAL = 1`.
SIMP exists to make a voxel-wise design choose between solid and void, and the decoders
already emit what is essentially a binary image: thresholding the converged density
changes the compliance by under a percent. The reference driver keeps `PENAL = 3`,
where every voxel is free and the penalization is what stops the optimizer from filling
the domain with grey.

The volume constraint itself is an augmented Lagrangian rather than a plain penalty. A
penalty alone has to grow without bound to close the gap, and a weight large enough to
do so swamps the compliance sensitivity: the volume then overshoots the target and
swings around it instead of settling. The driver therefore carries a multiplier
alongside the penalty, $w = \max(0, \lambda + c\,g)$ with $g = \bar\rho / V^* - 1$, and
integrates the multiplier slowly, $\lambda \leftarrow \max(0, \lambda + \eta\, g)$. The
proportional part reacts to a violation immediately, the multiplier accumulates the
weight the constraint actually needs, and clipping at zero keeps an inactive constraint
from pushing material back in. Both terms are small numbers: the compliance sensitivity
is normalized by the initial compliance, which puts the balanced weight at order one,
not at the hundreds the additive ramp used to reach.

Both decoders are trained on standardized images, so their output is undone with
`standardizer.inverse` and clamped to $[0, 1]$. The variational model is the two-stage
one, and the design variable is the code of its *second* stage, not the image code: the
second stage is fitted to the first stage's codes and its own prior is a standard
normal by construction, so the design can be initialized from a plain draw and its
rarity read off directly.

That is what the latent penalty measures, $w \cdot \tfrac{1}{2}\|z\|^2$, the negative
log density of the second-stage code under its prior up to a constant. The optimization
therefore returns a maximum a posteriori design: the most probable code that still
carries the load. The penalty is added before the gradient is clipped, so it competes
with the compliance sensitivity inside the same clipping budget rather than on top of
it.

A penalty on the log density pulls toward the mode, and the mode of a standard normal
is not a typical draw from it: the mass sits on a shell of radius $\sqrt{D}$, which the
diffusion driver below has to take seriously. Here it does no harm, because $D = 64$
and that shell is about $9\%$ wide. At $w = 10^{-3}$ the code comes to rest at $0.87$
of the shell radius, well inside the bulk, and the compliance is the same as for a
design whose code is held exactly on the shell. Only at $w \ge 10^{-1}$ does the mode
win: the code collapses to a fifth of the shell radius and the compliance rises by a
fifth with it. The same penalty in the $D = 65536$ of the diffusion model would be
ruinous at any weight, which is why that driver constrains the radius instead of
penalizing it.

The noise of a diffusion model is not a latent code of the same kind, and the penalty
that keeps the variational code probable has no counterpart here. The most probable
point of the standard normal the chain starts from is the origin, and the origin is not
a plausible draw: it denoises into the mean of the data, not into a microstructure. In
$D = 65536$ dimensions essentially all the mass of that normal sits in a thin shell at
radius $\sqrt{D}$, so the driver constrains rather than penalizes, $x_T = \sqrt{D}\, z
/ \|z\|$ with the direction $z$ as the design variable. This is the reparameterization
the penalty was reaching for: it costs no weight to tune, and the sampler is never
handed a noise level the model was not trained to denoise. A penalty
$w(\|x_T\|/\sqrt{D} - 1)^2$ on the free noise reaches the same shell, but only
approximately and with one more weight to balance against the compliance.

The direction is otherwise unconstrained, and `DRIFT_PENALTY` puts a trust region on
it, $w (1 - \cos(x_T, x_T^0))$ against the direction the run started from. It is
written on the cosine rather than on the angle because the derivative of $\arccos$ is
singular exactly where the run starts. Off by default: it buys manifold fidelity and
pays in compliance, monotonically. At $w = 0$ the design ends at compliance $31.2$ with
the holes merged into elongated voids and the direction $5 \cdot 10^{-2}$ radians from
its start; at $10^4$, $34.8$ and $1.5 \cdot 10^{-2}$ radians; at $3 \cdot 10^5$, $47.0$
and $3 \cdot 10^{-3}$ radians, with the sample still a set of strictly circular fibers.
Which end of that is wanted depends on whether the design or the microstructure is the
point.

Differentiating the sampler requires the deterministic variant: the stochastic sampler
of the training driver injects a fresh draw at every step, so its output is not a
function of $x_T$ alone. The chain is therefore a 25-step DDIM subsequence of the 200
trained steps. Each step is wrapped in `torch.utils.checkpoint`, as the activations of
25 u-net evaluations at $256^2$ do not fit at once, which costs one extra forward pass
per iteration. The clamp of the predicted clean image is kept in the forward pass but
skipped in the backward one (a straight-through estimator): towards the end of the
chain it saturates over most of the image and would otherwise zero the sensitivity
exactly where the design lives.

All three priors are left behind once the volume constraint bites: the circular fibers
grow and merge into elongated voids with a load path between them. How far the sampler
can be pushed decides how much of that happens. With the 25-step chain and a learning
rate of $10^{-3}$ the direction travels less than $10^{-2}$ radians over a whole run,
the samples stay strictly circular, and the compliance only tracks the volume, ending
near 48. Ten steps and $3 \cdot 10^{-3}$ move it by $5 \cdot 10^{-2}$ radians and reach
31.6, on par with the two autoencoders. The shorter chain is what makes the larger step
usable: a 25-step sample is sharper and its gradient correspondingly more brittle, so
the same learning rate collapses the volume instead of rearranging the holes. It is
also two and a half times cheaper per iteration, which pays for the longer run.

The learning rates are large for Adam ($1.5 \cdot 10^{-1}$ and $3 \cdot 10^{-1}$ on the
codes) because the design variable is a latent code, not a network weight: there are a
few hundred of them, they are updated a few hundred times, and each has to travel a
distance comparable to its own scale. At the original $5 \cdot 10^{-2}$ over 200
iterations none of the drivers finished moving, which is most of why their compliances
used to sit a third above the reference.
