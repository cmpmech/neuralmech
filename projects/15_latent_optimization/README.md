# Latent-space topology optimization

Topology optimization performed in the latent space of a generative model,
supporting **Chapter 15 (Generative Artificial Intelligence)**. The design is
decoded from a latent code rather than parameterized voxel-by-voxel, so every
candidate is constrained to lie on the learned manifold of fiber microstructures.

## Drivers

`topopt_latent_ae.py`
    Compliance minimization of a half MBB beam where the density field is the
    output of the fiber autoencoder (`../15_anomaly/fiber_ae_train.py`,
    `models/fiber_ae_1_256.pt2`). Adam optimizes the latent code, the volume
    constraint is a quadratic penalty, and the decoded fibers are read as holes so
    the surrounding matrix stays connected and load-bearing.

`topopt_latent_vae.py`
    The same optimization carried out in the latent space of the fiber variational
    autoencoder (`../15_anomaly/fiber_vae_train.py`,
    `models/fiber_vae_256_4.0_256.pt2`). The design variable is the code of the second
    stage, whose prior is a standard normal, and a penalty on its negative log density
    keeps the code probable, so the design stays typical of the training set.

`topopt_latent_diffusion.py`
    The same optimization inside the fiber diffusion model
    (`../15_anomaly/fiber_diffusion_train.py`, `models/fiber_diffusion_200_256.pt2`).
    The design variable is the noise the sampler starts from, and the deterministic
    reverse chain is differentiated end to end. The noise is constrained to the shell
    of the standard normal instead of being penalized towards its mode.

`topopt_reference.py`
    The classical density-based optimization the latent drivers are compared against:
    the same beam, discretization and volume fraction, but every voxel of the design
    grid is a design variable of its own, updated by the optimality criterion. Free of
    any learned manifold, it reaches a compliance of 28.9 against 31.4 for the
    variational autoencoder and 36.5 for the diffusion model.

## Non-obvious technicalities (authored by Claude)

The autoencoder only ever emits arrangements of circular fibers, which cover
roughly 4-28% of the domain. Read as material, those fibers would be disconnected
blobs with no load path, so the density is inverted (`rho = 1 - decoded`): the
fibers become voids punched out of a solid plate. A decoded training sample starts
around 0.89 material fraction; reaching the target of 0.6 forces the optimizer off
the strict fiber manifold, where the holes grow and merge into elongated voids.

The compliance sensitivity is computed on the design grid exactly as in a
classical solver, then handed to autograd as the incoming gradient of `rho`
(`rho.backward(sensitivity)`); backpropagation through the decoder turns it into a
gradient on the latent code. The volume penalty weight is ramped over the
iterations so the design first migrates to the target fraction and then rearranges
its holes toward the load path at (nearly) fixed volume.

The two decoders emit different quantities. The autoencoder is trained with a mean
squared error on standardized images, so its output is undone with
`standardizer.inverse`. The variational autoencoder is trained with a binary cross
entropy on logits, so its output passes through a sigmoid instead, which also bounds
the density to $(0, 1)$ without an explicit clamp. Only the latent mean is optimized;
sampling is switched off, so the design stays deterministic.

The latent penalty is $-w \log p(z)$ under the prior that the training driver fits to
the codes themselves. The optimization therefore returns a maximum a posteriori design:
the most probable code that still carries the load. The penalty is added before the
gradient is clipped, so it competes with the compliance sensitivity inside the same
clipping budget rather than on top of it.

Reading $\|z\|^2$ as the negative log density instead would require the codes to follow
a standard normal, and they do not: the trained prior scores a real code at $+664$ nats
and a draw from $\mathcal{N}(0, I)$ at $-1016$ nats, about $1700$ nats apart, so a
penalty on the norm would pull the design toward codes the decoder has never seen. The
learned density is far sharper than the quadratic it replaces, hence $w = 10^{-5}$
rather than $10^{-3}$; the two move the code by about the same amount per iteration.
Over a run the code still travels from $-644$ to $+765$ nats, because reaching the
volume target of $0.6$ means leaving the strict fiber manifold.

The noise of a diffusion model is not a latent code of the same kind, and the penalty
that keeps the variational code probable has no counterpart here. The most probable
point of the standard normal the chain starts from is the origin, and the origin is not
a plausible draw: it denoises into the mean of the data, not into a microstructure. In
$D = 65536$ dimensions essentially all the mass of that normal sits in a thin shell at
radius $\sqrt{D}$, so the driver constrains rather than penalizes,
$x_T = \sqrt{D}\, z / \|z\|$ with the direction $z$ as the design variable. This is
the reparameterization the penalty was reaching for: it costs no weight to tune, and
the sampler is never handed a noise level the model was not trained to denoise. A
penalty $w(\|x_T\|/\sqrt{D} - 1)^2$ on the free noise reaches the same shell, but only
approximately and with one more weight to balance against the compliance.

Differentiating the sampler requires the deterministic variant: the stochastic sampler
of the training driver injects a fresh draw at every step, so its output is not a
function of $x_T$ alone. The chain is therefore a 25-step DDIM subsequence of the 200
trained steps. Each step is wrapped in `torch.utils.checkpoint`, as the activations of
25 u-net evaluations at $256^2$ do not fit at once, which costs one extra forward pass
per iteration. The clamp of the predicted clean image is kept in the forward pass but
skipped in the backward one (a straight-through estimator): towards the end of the
chain it saturates over most of the image and would otherwise zero the sensitivity
exactly where the design lives.

The two priors constrain the design very differently. The autoencoder leaves its
manifold under the volume penalty -- the holes grow and merge into elongated voids --
while the diffusion sampler keeps emitting circular fibers and reaches the target
fraction by rearranging and adding holes instead. The sampler is also far more
sensitive to its input than the decoder is to a code: a learning rate of $5 \cdot
10^{-2}$, which the variational driver uses, wipes the fibers out within two
iterations, hence $10^{-3}$ here. The angle the direction travels over a whole run
stays below $10^{-2}$ radians.
