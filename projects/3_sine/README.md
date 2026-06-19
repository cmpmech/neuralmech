# Sine Fitting

Sine fitting toy problems for **Chapter 3 (Artificial Neural Networks)**. A 1D
$sin(2\pi x)$ target is the running example for regression, classification,
regularization, ensembling, Sobolev training, and the loss landscape visualizations (also used in **Chapter 2**).

## Data generation

- `sine_gen.py` -> `data/sine.npz`
  noisy sine, 64 samples
- `sobolev_sine_gen.py` -> `data/sobolev_sine.npz`
  sine **and** its derivative, 32 samples
- `discrete_sine_gen.py` -> `data/discrete_sine.npz`, `data/discrete_sine_test.npz` sine binned into 3 classes

## Drivers

- `mlp_sine_regression.py` _needs `sine.npz`_
  MLP regression on the sine
- `mlp_sine_classification.py` _needs `discrete_sine.npz`, `discrete_sine_test.npz`_
  MLP classifies the 3-class binned sine
- `mlp_sine_regularization.py` _needs `sine.npz`_
  weight decay, dropout, early stopping versus overfitting
- `mlp_sine_ensemble.py` _needs `sine.npz`_
  ensemble of MLPs for predictive uncertainty
- `mlp_sine_sobolev.py` _needs `sobolev_sine.npz`_
  Sobolev training: fit value and gradient jointly
- `mlp_sine_opt_landscape.py` _needs `sine.npz`_
  visualizes the (projected) loss landscape around the minimum
