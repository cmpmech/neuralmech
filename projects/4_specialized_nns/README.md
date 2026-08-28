# Specialized Neural Network Architectures

Specialized and advanced architectures for **Chapter 4 (Neural Network
Architectures)**. Each driver fits one architecture to a small shared toy: mostly
the noisy $\sin(2\pi x)$ and its derivative, the undamped harmonic oscillator, or
the Stanford bunny point cloud.

## Data generation

- `fno_sine_gen.py` -> `data/fno_sine_{32,64,128,256}.npz`
  noisy sine and its derivative sampled on grids of several resolutions
- `deeponet_sine_gen.py` -> `data/deeponet_sine_{32,64,128,256}.npz`
  shifted sines with a fixed sensor sampling and random query coordinates
- `hom_dmn_gen.py` -> `data/hom_dmn.npz`
  finite-cell homogenized effective stiffness of a two-phase circular-inclusion cell for
  many sampled phase moduli, plus a nonlinear-elastic FE reference of the same cell under a
  uniaxial macro-strain path (uses `solvers/homogenization.py` and the nonlinear-elastic
  cffi subroutine)
- `minecraft_mobs_download.py` -> `data/minecraft_mobs.npz`
  fixed-length mob-sound clips fetched from the minecraft wiki, labelled by behaviour class
  (neutral / passive / hostile) for the analog wave-network classifier; one mob per class is
  active by default (cow, creeper, enderman), audio kept flat in `external_data/minecraft_mobs/`

## Drivers

- `fno_example.py` _needs `fno_sine_*.npz`_
  Fourier neural operator mapping a function to its derivative across resolutions
- `fno_from_scratch.py`
  minimal 1D Fourier layer built from scratch (spectral convolution plus bypass)
- `cnn_neuraloperator.py` _needs `fno_sine_*.npz`_
  convolutional network used as a neural operator on the same task
- `deeponet_example.py` _needs `deeponet_sine_*.npz`_
  DeepONet with separate branch and trunk networks
- `elm_example.py` _needs `sine.npz`_
  extreme learning machine: closed-form least-squares readout on a fixed feature map
- `kan_example.py`
  Kolmogorov-Arnold network fitting a product of sines, with edge-strength diagnostics
- `siren_example.py`
  SIREN (sinusoidal activations) fitting a sharp signal and its gradient
- `siren_image_compression.py`
  SIREN fit to a photo at several hidden-layer widths, each giving a different
  raw-bytes-to-network-bytes compression ratio, compared against a jpeg saved at the
  matching byte budget
- `icnn_example.py`
  input-convex neural network fitting a parabola
- `icnn_neuralode_example.py`
  ICNN as the right-hand side of a neural ODE
- `hnn_example.py`
  Hamiltonian neural network learning the energy of a harmonic oscillator
- `lnn_example.py`
  Lagrangian neural network learning the Lagrangian of the same system
- `mlp_dynamics_example.py`
  plain MLP baseline predicting the same dynamics directly
- `dmn_example.py` _needs `hom_dmn.npz`_
  deep material network: a laminate-tree topology (the batched DMN forward in `NN.py`)
  fit to the finite-cell effective stiffness, then frozen-tangent nonlinear-elastic
  prediction with no retraining, checked against the finite-cell reference generated
  alongside the dataset
- `gnn_cheb_example.py` _needs `ghana_mesh.pt`_
  Chebyshev spectral graph convolution regressing a scalar field on the Ghana mesh
- `pointnet_example.py` _needs `bunny_pointcloud.npz`_
  PointNet regressing a scalar field on the Stanford bunny point cloud
- `pointnetpp_example.py` _needs `bunny_pointcloud.npz`_
  PointNet++ (hierarchical set abstraction) on the same point cloud
- `analog_rnn_train.py` _needs `minecraft_mobs.npz`_
  analog recurrent network (Hughes et al. 2019): a trainable acoustic medium that classifies
  mob-sound clips by the wave energy each reaches at three right-wall probes, one per behaviour
  class, trained end-to-end through cuwave's boundary-reconstruction adjoint (solvers/cuwave);
  saves the binarized medium to `models/analog_rnn_material.npy`
- `analog_rnn_eval.py` _needs `analog_rnn_material.npy`_
  renders a wavefield snapshot plus source and probe signals for one clip propagated
  through the trained medium (or the free field)
- `analog_rnn_fixture.py`
  shared acoustic setup (grid, sponge, source/probe positions, clip band-compression) plus
  the probe readout and its cross-entropy objective, imported by the analog RNN train and
  eval drivers

## Non-obvious technicalities (authored by Claude)

`analog_rnn_train.py` runs one full wave solve (forward and adjoint) per clip, so an epoch
over all 53 clips is the runtime bottleneck. When first exploring the setup -- resolution,
propagation time `T`, material contrast, low-pass cutoff -- start with a single clip per class
(three solves per epoch instead of 53). This cuts iteration time by more than an order of
magnitude and already answers the key question, whether the trainable medium can route the
three mobs to their probes, before scaling up to the full imbalanced set. With one clip per
class the data are balanced, so the inverse-frequency class weighting is a no-op there.
Subsample right after the npz is loaded, keeping the first index of each class.
