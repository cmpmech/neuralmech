# Optimization Techniques

Worked examples for **Chapter 2 (Fundamental Machine Learning)**, covering both gradient-based and gradient-free optimization on two standard test functions: the Rosenbrock valley and the multimodal Ackley function.

## Drivers

- `optimization_config.py`
  shared test functions (Rosenbrock, Ackley) and a plot of the Ackley landscape
- `gradient_optimizers.py`
  trajectories of steepest descent, momentum, Adagrad, RMSprop, Adam, and L-BFGS on
  the Rosenbrock function
- `ackley_adam.py`
  Adam on the multimodal Ackley function, trapped in a local minimum
- `adam_from_scratch.py`
  Adam implemented from scratch on a 1D quartic
- `lbfgs_from_scratch.py`
  L-BFGS two-loop recursion implemented from scratch on a 1D quartic
- `param_search.py`
  structured grid search versus random sampling on the Ackley function
- `genetic_algorithm.py`
  elitist genetic algorithm on the Ackley function
- `particle_swarm_optimizer.py`
  particle swarm optimization on the Ackley function
- `bayesian_optimizer.py`
  Bayesian optimization with a Gaussian-process surrogate on the Ackley function
