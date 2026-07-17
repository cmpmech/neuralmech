# Latent-space topology optimization

Topology optimization performed in the latent space of a generative model,
supporting **Chapter 15 (Generative Artificial Intelligence)**. The design is
decoded from a latent code rather than parameterized voxel-by-voxel, so every
candidate is constrained to lie on the learned manifold of fiber microstructures.

## Drivers

`topopt_latent.py`
    Compliance minimization of a half MBB beam where the density field is the
    output of the fiber autoencoder (`../15_anomaly/fiber_ae_train.py`,
    `models/fiber_ae_1_256.pt2`). Adam optimizes the latent code, the volume
    constraint is a quadratic penalty, and the decoded fibers are read as holes so
    the surrounding matrix stays connected and load-bearing.

## Non-obvious technicalities (authored by Claude)

The autoencoder only ever emits arrangements of circular fibers, which cover
roughly 5-22% of the domain. Read as material, those fibers would be disconnected
blobs with no load path, so the density is inverted (`rho = 1 - decoded`): the
fibers become voids punched out of a solid plate. The reachable material fraction
is therefore about 0.78-0.95, which is why the volume target sits near the top of
that band rather than at a conventional 0.5.

The compliance sensitivity is computed on the design grid exactly as in a
classical solver, then handed to autograd as the incoming gradient of `rho`
(`rho.backward(sensitivity)`); backpropagation through the decoder turns it into a
gradient on the latent code. The volume penalty weight is ramped over the
iterations so the design first migrates to the target fraction and then rearranges
its holes toward the load path at (nearly) fixed volume.
