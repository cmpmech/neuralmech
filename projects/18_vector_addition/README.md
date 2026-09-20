# Vector Addition on the GPU

The smallest possible CUDA example for **Chapter 18 (Simulation Acceleration via
GPUs)**: adding two vectors of a million entries.

## Drivers

- `add.py`
  launches the kernel through cupy and times it; `PARALLEL` switches between the
  single-thread loop and the one-thread-per-entry version in `add.cu`
