import numpy as np

np.random.seed(0)
f = lambda x: np.sin(2 * np.pi * x)
samples = 64
noise = 0.2

X = np.random.uniform(-1, 1, (samples, 1))

Y = f(X) + noise * np.random.uniform(-1, 1, (samples, 1))

np.savez(f'../../data/sine.npz', X=X, Y=Y)