# Deep Learning Beyond PyTorch

The sine regression MLP of **Chapter 3 (Artificial Neural Networks)** in other frameworks and
languages, for **Appendix B (Deep Learning Beyond PyTorch)**. Every driver generates the same
kind of noisy sine (64 samples) inline, trains a $[1, 24, 24, 24, 1]$ tanh MLP with Adam for
2000 full-batch epochs, and prints the final cost.

## Drivers

- `mlp_sine_keras.py`
  Keras (TensorFlow backend), `python mlp_sine_keras.py`
- `mlp_sine_jax.py`
  JAX with optax, `python mlp_sine_jax.py`
- `mlp_sine.jl`
  Julia with Lux, Zygote and Optimisers, `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
  once, then `julia --project=. mlp_sine.jl`
- `mlp_sine.cpp`, `CMakeLists.txt`
  C++ with LibTorch, the C++ API of PyTorch; build against the libtorch inside the installed
  torch: `cmake -S . -B build -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')`,
  `cmake --build build`, `./build/mlp_sine`
- `rust/`
  Rust with Burn, `cd rust && cargo run --release`
- `mlp_sine_export.py` -> `models/mlp_sine_weights.txt`
  trains the MLP in PyTorch and writes its weights as text, column-major per layer
- `mlp_sine_forward.f90` _needs `mlp_sine_weights.txt`_
  forward pass in plain Fortran, as it would sit inside a solver (e.g., a UMAT);
  `gfortran mlp_sine_forward.f90 -o mlp_sine_forward && ./mlp_sine_forward`, prints the same
  prediction as the export script
