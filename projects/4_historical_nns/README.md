# Historical Neural Networks

Minimal implementations of historical networks for **Chapter 4 (Neural Network
Architectures)**. Each driver is self-contained and runs on the CPU in seconds, on toy data:
logic gates, random binary patterns, bars-and-stripes images, and the sine.

## Drivers

- `perceptron.py`
  Rosenblatt's learning rule on AND (converges) and XOR (never converges)
- `hopfield_net.py`
  Hebbian storage of random patterns and recall from a corrupted one
- `som.py`
  self-organizing map unfolding a 2D grid onto a ring of data
- `restricted_boltzmann.py`
  restricted Boltzmann machine trained by contrastive divergence, sampled by Gibbs sampling
- `deep_belief_net.py`
  greedy layer-wise RBM pretraining, then backpropagation fine-tuning of a bars/stripes classifier
- `rbf_net.py` _needs `sine.npz`_
  radial basis function network with k-means centers and a closed-form readout
- `reservoir.py`
  echo state network with a ridge-regression readout, forecasting a sum of sines autonomously
- `helper.py`
  bars-and-stripes data and the contrastive divergence step shared by the two RBM drivers
