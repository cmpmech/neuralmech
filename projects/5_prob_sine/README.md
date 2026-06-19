# Probabilistic Deep Learning

Toy problems for **Chapter 5 (Probabilistic Deep Learning)**, on equipping
predictions with uncertainty. Two targets recur: a 1D Gaussian for the sampling
demonstrations, and the noisy $\sin$ from Chapter 3 for the regression models.

## Drivers

- `mcmc.py`
  Metropolis-Hastings sampling of a 1D Gaussian target
- `hmc.py`
  Hamiltonian Monte Carlo sampling of the same target, by hand
- `hmc_pymc.py`
  the same Hamiltonian Monte Carlo sampling via PyMC, as a reference
- `mle_homo_sine.py`
  maximum-likelihood regression with a single learned (homoscedastic) noise variance
- `mle_hetero_sine.py`
  maximum-likelihood regression with input-dependent (heteroscedastic) variance
- `bayesian_hmc_sine.py`
  Bayesian neural network posterior sampled with Hamiltonian Monte Carlo (hamiltorch)
- `bayesian_vainf_sine.py`
  Bayesian neural network trained by mean-field variational inference
