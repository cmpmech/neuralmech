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
    The same optimization carried out on the latent mean of the fiber variational
    autoencoder (`../15_anomaly/fiber_vae_train.py`,
    `models/fiber_vae_260_1.0_256.pt2`). A penalty on the negative log density under the
    learned prior keeps the code probable, so the design stays typical of the training
    set.

`test_vae.py`
    Decodes one code drawn from a standard normal next to one drawn from the learned
    prior, which tests whether the model generates rather than only reconstructs.

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
