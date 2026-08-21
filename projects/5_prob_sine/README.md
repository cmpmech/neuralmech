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
- `bayesian_ale_vainf_sine.py`
  cooperative three-step training (mean, variance, Bayesian network) that separates
  aleatoric from epistemic uncertainty, after Yi & Bessa (arXiv:2505.02743)
- `bayesian_ale_hmc_sine.py`
  the same three-step separation, with the posterior sampled by Hamiltonian Monte Carlo
  instead of variational inference
- `bayesian_ale_hmc_sineV2.py`
  the same, but with the variance network fitted by the plain Gaussian likelihood of the
  squared residual instead of the gamma reformulation, as an ablation of that trick
- `active_bayesian_sine.py`
  active learning, where the next sample is measured wherever the Bayesian posterior
  is least certain, starting from 5 samples in $[-1, 1]$
- `active_random_sine.py`
  the same fit sampling at random, as the baseline the active strategy has to beat
