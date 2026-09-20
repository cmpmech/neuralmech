# Neural Reparametrization

Optimization where the design variable is not the quantity itself but the weights of
a network that emits it, supporting **Chapter 16 (Neural Optimization)**. The
reparametrized problem has a different, usually easier, loss landscape than the one
written directly in the design variables.

## Drivers

- `reparametrization1D.py`
  a single scalar minimized on a rough 1D objective, directly and through a network,
  with the spread over initializations showing how often each ends up in the global
  minimum
- `reparametrization2D.py`
  the same comparison on the standard 2D benchmarks (rosenbrock, rastrigin, ackley,
  levy), with the optimizer paths drawn over the objective
- `neuraltopopt_mbb.py`
  compliance minimization of a half MBB beam where the density field is the output of
  a convolutional generator, a multilayer perceptron, or the voxels themselves
