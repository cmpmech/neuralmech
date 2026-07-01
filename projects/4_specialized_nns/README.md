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
- `dmn_data.py` -> `data/dmn_dataset.npz`
  finite-cell homogenized effective stiffness of a two-phase circular-inclusion cell
  for many sampled phase moduli (uses `solvers/homogenization.py`)

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
- `dmn_example.py` _needs `dmn_dataset.npz`_
  deep material network: a laminate-tree topology fit to the finite-cell effective
  stiffness, then nonlinear-elastic prediction with no retraining, checked against a
  finite-cell reference solved with the same material law
- `pointnet_example.py` _needs `bunny_pointcloud.npz`_
  PointNet regressing a scalar field on the Stanford bunny point cloud
- `pointnetpp_example.py` _needs `bunny_pointcloud.npz`_
  PointNet++ (hierarchical set abstraction) on the same point cloud
