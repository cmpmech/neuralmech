# Limitations of Deep Learning

Illustrates limitations of state-of-the-art deep learning for **Chapter 1 (Computational Mechanics Meets Artificial Intelligence)**.

## Data generation

- `vlm_probes.py` -> `results/limitations_probes/`
  test images with known answers for multimodal models (mazes, analog clocks, overlapping
  circles, counting, a modified Muller-Lyer illusion, a prompt injection), questions and
  answers in `answers.json`; 1600 px PNGs to upload to the models, vector PDFs
  (`results/rgb_pdf/vlm_*.pdf`) for the book

## Drivers

- `sine_mlp_sensitivity.py` _needs `sine.npz`_
  sweeps initialization seed against learning rate and records the final validation
  error as a heatmap
- `trainability_fractal.py`
  sweeps the learning rates of both layers of a small tanh network and zooms into the
  fractal boundary between converging and diverging training
- `sine_extrapolation.py` _needs `sine.npz`_
  MLPs from several seeds agree inside the training interval and disagree outside
- `rotation_shift.py`
  probability of the correct class of a pretrained EfficientNetV2 under image rotation, with
  the top-1 class every 45 degrees
- `adversarial_perturbation.py`
  targeted projected gradient descent turns a pretrained EfficientNetV2 prediction into any
  chosen class with an invisible perturbation
- `shortcut_learning.py`
  colored MNIST: an MLP learns the color instead of the digit and fails once colors are random
- `random_labels.py`
  an MLP memorizes MNIST with random labels as perfectly as with the true ones
- `catastrophic_forgetting.py` _needs `sine.npz`_
  an MLP trained on the left half of the sine data, then on the right half, forgets the left
- `rollout_drift.py`
  an MLP learns one time step of a pendulum; its autoregressive rollout loses energy
- `one_pixel_attack.py`
  exhaustive search for single pixels that flip the prediction of an MLP on MNIST digits
- `fooling_images.py`
  targeted gradient ascent turns uniform noise into any class with 100 % probability of EfficientNetV2
- `translation_shift.py`
  MNIST test accuracy of an MLP and a CNN under horizontal shifts of the digits
- `saliency_maps.py`
  SmoothGrad saliency of EfficientNetV2 for a goose and for its adversarial toaster look alike

## Non-obvious technicalities (authored by Claude)

- `trainability_fractal.py` trains all $512^2$ networks at once with hand-written gradients
  (batched `einsum`), which is far faster than `vmap` over autograd for this tiny network.
  Data and initialization are identical for every pixel; only the two learning rates differ.
  Converged runs are colored by their mean cost, diverged runs by how fast they diverged (the
  coloring of Sohl-Dickstein, https://arxiv.org/abs/2402.06184). Each zoom centers on the
  boundary point with the most intertwined converged and diverged pixels.
- `adversarial_perturbation.py` attacks in pixel space $[0, 1]$ and applies the ImageNet
  normalization inside the model, so the budget `EPSILON = 1/255` is exactly one gray level.
- `shortcut_learning.py` needs a perfect color-label correlation in training. At 99 % the
  network also picks up the digit shapes and still reaches about 64 % test accuracy.
- `rollout_drift.py` uses the dimensionless pendulum $\ddot q = -\sin q$ with state $(q, p)$,
  $p = \dot q$, written $\varphi$ and $\dot\varphi$ in the book. The reference step is a
  fourth-order Runge-Kutta scheme with 20 substeps, exact to plotting accuracy. The network
  predicts the rate $(s_{\tau+1} - s_\tau)/\Delta t$, which keeps its output of order one.
  Other settings (e.g. `Q0 = 2.0`, training range 2.5) make the energy drift up instead; the sign is
  arbitrary, the drift is not.
- `one_pixel_attack.py` tries all $2 \times 28 \times 28$ single-pixel changes (black or white)
  per digit in one batch, so no evolutionary search is needed. At the $480^2$ input of
  EfficientNetV2 the same search would be far too expensive.
- The small CPU drivers here slow down by a factor of about 40 when the machine is busy and
  torch uses all cores; `OMP_NUM_THREADS=4` avoids it.
- `saliency_maps.py` takes the largest gradient magnitude over the color channels per pixel and
  clips the map at its 99th percentile before normalizing, otherwise a few pixels saturate the
  colormap. A randomly initialized EfficientNetV2 (the sanity check of Adebayo et al.,
  https://arxiv.org/abs/1810.03292) gave a blurry map that does not resemble the goose, so it is
  not shown.
