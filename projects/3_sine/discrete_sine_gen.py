import numpy as np

np.random.seed(1)
f = lambda x: np.sin(2 * np.pi * x)

def discretize(y, classes):
    thresholds = np.linspace(-1, 1, classes + 1)
    y_ = np.zeros_like(y)
    for threshold in thresholds[1:-1]:
        y_[y > threshold] += 1
    return y_.astype(np.long)

samples = 128
classes = 3
noise = 0.

# ---------------------- training & validation data ----------------------
X = np.random.uniform(-1, 1, (samples, 1))
Y = discretize(f(X) + noise * np.random.uniform(-1, 1, (samples, 1)),
               classes)
np.savez(f'../../data/discrete_sine.npz', X=X, Y=Y.squeeze())

# ----------------------------- testing data -----------------------------
X = np.expand_dims(np.linspace(-1.3, 1.3, 1000), 1)
Y = discretize(f(X) + noise * np.random.uniform(-1, 1, (1000, 1)),
               classes)
np.savez(f'../../data/discrete_sine_test.npz', X=X, Y=Y.squeeze())