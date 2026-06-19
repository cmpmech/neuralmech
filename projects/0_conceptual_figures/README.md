# Conceptual Figures

Conceptual figures used throughout the entire book. Each script is self-contained
and produces one schematic illustration. The numeric prefix is the chapter the
figure appears in.

## Chapter 1 (Computational Mechanics Meets Artificial Intelligence)

- `1_imagenet.py`
  a grid of sample ImageNet photographs
- `1_mnist.py`
  a grid of sample MNIST digits
- `1_weather_pressure.py`
  ERA5 mean-sea-level pressure on a (Mollweide) world map
- `1_weather_temperature.py`
  ERA5 2 m air temperature on a (Mollweide) world map
- `1_weather_wind.py`
  ERA5 wind field as streamlines on a (Mollweide) world map

## Chapter 2 (Fundamental Machine Learning)

- `2_adam.py`
  momentum and AdaGrad optimizer trajectories on a 2D loss surface
- `2_ml_tasks.py`
  the four machine-learning task types: regression, classification,
  representation learning, generative modeling

## Chapter 3 (Artificial Neural Networks)

- `3_data_augmentation.py`
  common image data-augmentation transforms applied to one photograph
- `3_gabor_regularized.py`
  a Gabor-fit loss landscape with and without regularization, and their minima
- `3_mlp_expressivity_learn.py`
  an MLP learning to reproduce an image, showing how expressivity grows with
  width (animated)
- `3_softmax.py`
  softmax turning sampled logits into a probability distribution
- `3_universal_approximation_depth_comb.py`,
  `3_universal_approximation_depth_layer1.py`,
  `3_universal_approximation_depth_layer2.py`
  universal approximation by hand-set ReLU weights: the per-layer contributions
  (`layer1`, `layer2`) and the combined deep network (`comb`)

## Chapter 4 (Neural Network Architectures)

- `4_bspline.py`
  a quadratic B-spline curve and its basis functions
- `4_graph_concept.py`
  a Delaunay graph over scattered points, illustrating the graph-network setting

## Chapter 5 (Probabilistic Deep Learning)

- `5_wasserstein.py`
  optimal transport between two distributions and the Wasserstein distance,
  solved as a linear program

## Chapter 15 (Generative Artificial Intelligence)

- `15_noising.py`
  the forward diffusion process progressively noising an image
