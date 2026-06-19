# Multilayer Perceptrons

Toy problems for **Chapter 3 (Artificial Neural Networks)** covering multilayer perceptron fundamentals.

## Drivers

- `ad.py`
  reverse-mode automatic differentiation engine from scratch (a scalar computation
  graph), evaluated on $d = (ab + c)\,a$
- `ad_torch.py`
  the same example through PyTorch autograd, validating `ad.py`
- `mlp_from_scratch.py`
  MLP forward propagation and backpropagation in numpy, fitting $y = x^2$ and
  validated against a weight-matched PyTorch model
- `universal_approx_relu.py`
  visual universal approximation: a two-layer ReLU MLP fits $\sin(\pi x)$ by spreading
  equidistant kinks and solving the output layer by least squares
- `mlp_expressivity.py`
  expressivity of depth versus width, visualized by mapping 2D coordinates
  $\mathbf{x} \in [-1, 1]^2$ to RGB images with randomly initialized MLPs
- `gabor_sgd.py`
  stochastic gradient descent fitting a Gabor function; visualizes the loss
  landscape and optimizer trajectories for different batch sizes
