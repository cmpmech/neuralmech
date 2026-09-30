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
- `neuralfwi_data.py -> data/neuralfwi.npz`
  random elliptical voids with the first FWI gradient of each (homogeneous start)
- `neuralfwi_pretraining.py -> models/neuralfwi_unet.pt`
  a U-Net pretrained from the first gradient to the void indicator, _needs neuralfwi.npz_
- `neuralfwi.py`
  synthetic full waveform inversion (cuwave) of a circle stack on the voxels, through a
  smoothed Heaviside projection, through a
  randomly initialized generator, or through the pretrained U-Net (transfer learning),
  _needs neuralfwi_unet.pt_ for `--ansatz pretrained`
- `neuralfwi_wave.py`
  one shot through the true geometry of `neuralfwi.py`, as a video of the wavefield
