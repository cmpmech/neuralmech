# GPUs for Deep Learning

Timing sweeps of CPU against GPU for **Chapter 18 (Simulation Acceleration via GPUs)**,
section "GPUs for Deep Learning". With `--book`, each driver writes `results/data/*.csv`
with columns `x` (sweep variable), `y1` (GPU time in s) and `y2` (CPU time in s);
copy them to `book/plottingdata/`.

- `matrix_matrix.py` -> `matrix_matrix.csv` (`matrix_matrix_gpu.tex`)
  `N x N` matrix-matrix product, cuBLAS (cupy) against BLAS (numpy)
- `fnn.py` -> `fnn_<N>_<grad>.csv` (`fnn_gpu.tex`)
  one linear layer `N -> N` over the batch size, forward only and forward plus backward (PyTorch)
- `sparse_matrix_vector.py` -> `sparse_matrix_vector_<s>_<mkl>.csv` (`matrix_vector_sparse_gpu.tex`)
  sparse matrix-vector product for four sparsity levels; run once plain (BLAS, single
  precision) and once with `--mkl` (MKL, double precision, needs `sparse_dot_mkl`)
- `convolutions.py` -> `conv_<k>.csv`
  `k x k` correlation of an `N x N` image, cupyx against scipy, for `k` = 1, 3, 5, 7
- `matmul.cu`
  naive matrix-matrix kernel, one thread per entry of the result (listing in the chapter)
- `conv2D.cu`
  direct `k x k` correlation, one thread per output pixel (listing in the chapter)
