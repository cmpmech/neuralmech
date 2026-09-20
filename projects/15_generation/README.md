# Generative Models of Shapes

Generative models trained on a set of six binary shapes, supporting **Chapter 15
(Generative Artificial Intelligence)**. The same dataset runs through an
autoencoder, a variational autoencoder, an adversarial pair and a diffusion model,
so the latent spaces and the samples can be compared side by side.

## Data generation

- `shape_gen.py` -> `data/shapes_<label>_<R>.npy`
  circles, squares, triangles, ellipses, stars and crosses at random positions

## Drivers

- `shape_ae_train.py` _needs `shapes_<label>_<R>.npy`_
  convolutional autoencoder on the pooled shapes
- `shape_ae_latent.py`
  encodes every class into the 2D latent plane and decodes a grid of points from it,
  showing the gaps a plain autoencoder leaves between the classes
- `shape_vae_train.py` _needs `shapes_<label>_<R>.npy`_
  the variational counterpart, with `beta` trading reconstruction against a tidy
  latent space
- `shape_vae_latent.py`
  the same plane for the variational model, where the classes fill it without gaps
- `shape_vae_high_latent.py`
  walks the diagonal of a higher-dimensional latent cube, where interpolation stays
  on the shape manifold
- `shape_GAN.py` _needs `shapes_<label>_<R>.npy`_
  unconditional generative adversarial network, with the mode spread tracked over
  training as a collapse indicator
- `shape_WGAN.py` _needs `shapes_<label>_<R>.npy`_
  the Wasserstein variant with a gradient penalty, whose critic cost tracks sample
  quality
- `shape_diffusion.py` _needs `shapes_<label>_<R>.npy`_
  denoising diffusion model with a cosine noise schedule

## Non-obvious technicalities (authored by Claude)

The generators upsample with a learned `ConvTranspose2d` in the `pre_modules` slot
rather than a fixed `nn.Upsample`. Nearest-neighbour interpolation followed by $3
\times 3$ convolutions is a low-pass pipeline, and it rounds off exactly the corners
that separate a square from a blob. Kernel 2 with stride 2 divides evenly, so the
transposed convolution never overlaps itself and the checkerboard artefact it is
usually blamed for cannot arise.

The adversarial pair differ in one structural place: the discriminator of
`shape_GAN.py` keeps batch normalization, the critic of `shape_WGAN.py` has no
normalization at all. The gradient penalty is taken per sample, and batch
normalization couples the samples within a batch, which invalidates it.

Both drivers track a mode spread, the mean pairwise distance between generated
images. It is the collapse indicator the adversarial costs do not give: a collapsed
generator returns nearly the same picture whatever the latent, driving the spread
towards zero while both costs stay unremarkable. The Wasserstein critic cost, in
contrast, does track sample quality.

The diffusion sampler of `shape_diffusion.py` steps through the predicted clean
image, clipped to $[-1, 1]$, rather than applying the plain update $x - \beta /
\sqrt{1 - \bar\alpha}\,\epsilon$. That update divides by $\sqrt{\alpha}$, and the
near-unit $\beta$ at the end of the cosine schedule blow it up into all-black or
all-white images.
