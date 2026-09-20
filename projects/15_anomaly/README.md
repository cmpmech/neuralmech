# Anomaly Detection

Unsupervised anomaly detection for **Chapter 15 (Generative Artificial
Intelligence)**: a generative model is fitted to healthy data alone, and anything it
reconstructs badly is flagged. Two datasets carry the idea, a 3-dof oscillator signal
and a fiber-reinforced microstructure.

## Data generation

- `signal_normal_gen.py` -> `data/normal_3dof_<N>.npy`
  healthy responses of randomly drawn 3-dof oscillators
- `signal_k_gen.py` -> `data/anomaly_3dof_k.npz`
  the same oscillator with a stiffness that degrades over a bump, swept over bump
  width and depth
- `signal_f_gen.py` -> `data/anomaly_3dof_amp.npz`
  the same oscillator with a forcing spike, swept over spike width and height
- `fiber_gen.py` -> `data/fibers_<R>.npy`, `data/fibers_anomaly_<k>_<R>.npy`,
  `data/fibers_anomaly_structured_<R>.npy`
  randomly packed circular fibers, the same packings with `k` fibers replaced by
  squares, and a regular grid of fibers

`signal_config.py` holds the shared base problem and `signal_helper.py` the
degradation and spike laws the two anomaly generators apply.

## Drivers

- `fiber_ae_train.py` _needs `fibers_<R>.npy`_
  convolutional autoencoder fitted to healthy microstructures
- `fiber_ae_detect_anomaly.py` _needs `fibers_anomaly_<k>_<R>.npy`_
  reconstruction error against the number of square inclusions, the detection score
- `fiber_ae_detect_anomaly_structured.py` _needs `fibers_anomaly_structured_<R>.npy`_
  the same autoencoder on a regular grid, an arrangement it never saw
- `fiber_vae_train.py` _needs `fibers_<R>.npy`_
  variational autoencoder with a second-stage prior fitted to the encoded codes
- `fiber_vae_sample.py`
  draws microstructures from that learned prior
- `fiber_diffusion_train.py` _needs `fibers_<R>.npy`_
  denoising diffusion model on the same microstructures
- `fiber_diffusion_sample.py`
  runs the reverse chain of the trained diffusion model

`slot_vae.py` is a shelved reference, not a driver: it records why a slot-structured
latent reconstructs better and still cannot be sampled from a standard normal prior.
