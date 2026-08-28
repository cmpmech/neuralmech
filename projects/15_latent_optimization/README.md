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
    `models/fiber_vae_64_3.0_256.pt2`). A quadratic penalty holds the code on the shell
    that the code stays probable under the prior of the variational autoencoder, so the
    design stays typical of the training set.

`test_vae.py`
    Decodes a single code drawn from the prior of the variational autoencoder, which
    tests whether the model generates rather than only reconstructs.

## Non-obvious technicalities (authored by Claude)

The autoencoder only ever emits arrangements of circular fibers, which cover
roughly 5-22% of the domain. Read as material, those fibers would be disconnected
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

The latent penalty is $w \|z\|^2$, which for a latent that follows a standard normal
distribution is its negative log density up to a constant. The optimization therefore
returns a maximum a posteriori design: the most probable code that still carries the
load. The penalty is added before the gradient is clipped, so it competes with the
compliance sensitivity inside the same clipping budget rather than on top of it.

The interpretation depends on the latent actually matching the prior, which is what
`BETA` in the training driver buys. It also assumes the penalty stays weak: a normal
distribution in $H$ dimensions carries its mass on a shell of radius $\sqrt{H}$ rather
than at the origin, so a strong penalty pulls the code inside that shell and decodes
toward a blurred average microstructure. With $w = 10^{-3}$ the code moves from $8.5$ to
$7.1$ against a shell radius of $8$, which is about one standard deviation of the radius
and still an ordinary code.
