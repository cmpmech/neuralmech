# Vector Addition on the GPU

The smallest possible CUDA examples for **Chapter 18 (Simulation Acceleration via
GPUs)**: adding two vectors on one GPU, and adding them or taking their dot product
on several.

## Drivers

- `add.py`
  launches the kernel through cupy and times it; `PARALLEL` switches between the
  single-thread loop and the one-thread-per-entry version in `add.cu`
- `add_multi_gpu.py`
  `addParallel` on several GPUs, each adding its own chunk of the vectors without
  exchanging data
- `dot_multi_gpu.py`
  the dot product on several GPUs, each summing its own chunk with `cp.dot` before
  the host adds the partial sums
