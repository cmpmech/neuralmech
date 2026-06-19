# Neural Network Architectures

A tour of the architectures in **Chapter 4 (Neural Network Architectures)**. Each
driver poses similarly simple regression tasks to a different architecture, so the
inductive bias of the architecture is what stands out: feedforward, convolutional,
graph, recurrent, and continuous-depth networks, plus the building blocks (skip
connections, normalization, parameter sharing) that make deep networks trainable.

## Drivers

- `mlp_example.py`
  multilayer perceptron mapping vectors of random noise to a 1D sinusoid
- `cnn_example.py`
  convolutional network mapping images of random noise to a 2D sinusoidal field
- `cnn_filters.py`
  fixed $3\times 3$ kernels (edge detection, embossing, Gaussian blur, ...) applied
  to an image, illustrating what the learnable filters of a CNN can represent
- `transposed_conv.py`
  checkerboarding artifacts from a transposed convolution when the kernel size is
  not divisible by the stride ($K = 3$, $s = 2$)
- `gnn_example.py`
  GraphSAGE message-passing network mapping random node noise to a sinusoid on an
  unstructured triangular mesh
- `gnn_gin_example.py`
  the same graph task with a graph isomorphism network (GIN)
- `rnn_example.py`
  RNN, LSTM, and GRU on a sequence regression task: generate a sum of sines from a
  constant input triggered only by an initial impulse
- `tcn_example.py`
  temporal convolutional network on the same sequence task, showing how a receptive
  field smaller than the sequence length fails
- `resnet_mlp.py`
  effect of skip connections and weight initialization on the average gradient
  magnitude per layer in a 40-layer MLP (vanishing gradients)
- `normalization_mlp.py`
  effect of batch and layer normalization on the average activation magnitude per
  layer in a deep MLP
- `param_sharing.py`
  imposing structure on the network to learn the symmetric function
  $y = \sin(2\pi x_1)\sin(2\pi x_2)$ from few samples: a plain MLP, two separate
  networks combined multiplicatively (separability), and identical networks with
  shared parameters (symmetry)
- `neuralode_example.py`
  neural ODE regressing $y(\tau) = \sin(2\pi\tau)$ from sparse, irregularly sampled
  observations and an initial condition
- `neuralaugode_example.py`
  augmented neural ODE on the same task, reaching similar results faster by
  inflating the state with an extra latent channel
