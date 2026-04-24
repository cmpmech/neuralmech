from pathlib import Path

import torch
import numpy as np
from torch.utils.data import TensorDataset
from postprocessing import save_csv
import matplotlib.pyplot as plt
import tntorch
import time

BASE_DIR = Path(__file__).parent
torch.manual_seed(0)

# ---------------------------- model settings ----------------------------
# p controls the polynomial degree (hyperbolic truncation threshold);
# higher p = richer basis = better fit but slower and more prone to diverge
p = 8

# ----------------------------- prepare data -----------------------------
data = np.load(BASE_DIR / '../../data/sine.npz')
dataset = TensorDataset(torch.from_numpy(data['X']).to(torch.float32),
                        torch.from_numpy(data['Y']).to(torch.float32))
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])

X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices].squeeze()
X_val   = train_data.dataset.tensors[0][val_data.indices]
Y_val   = train_data.dataset.tensors[1][val_data.indices].squeeze()

# --------------------------------- fit ----------------------------------
# PCEInterpolator learns empirical orthogonal polynomials from the input
# distribution and fits a sparse polynomial chaos expansion via LARS.
# This is a closed-form step (no gradient descent).
model = tntorch.PCEInterpolator()
tic = time.time()
model.fit(X_train, Y_train, p=p, verbose=True)
toc = time.time()
print(f'elapsed fitting time: {toc - tic:.2e}s')

# ---------------------------- postprocessing ----------------------------
y_val_pred = model.predict(X_val)
cost = float(((y_val_pred - Y_val) ** 2).mean())
print(f'validation cost: {cost:.2e}')

f = lambda x: torch.sin(2 * torch.pi * x)
x_test = torch.linspace(-1.3, 1.3, 100).unsqueeze(1)
y_test = f(x_test.squeeze())

y_pred = model.predict(x_test)

fig, ax = plt.subplots()
ax.plot(x_test.squeeze(), y_test, 'k')
ax.plot(x_test.squeeze(), y_pred, 'r--')
ax.plot(X_train, Y_train, 'bo')
ax.set_ylim(-2, 2)
plt.show()

# ------------------------- book postprocessing --------------------------
save_csv(BASE_DIR / '../../results/tt_sine_test.csv',
         x=x_test[:, 0], y=y_test, ypred=y_pred)
save_csv(BASE_DIR / '../../results/tt_sine_train.csv',
         x=X_train[:, 0], y=Y_train)
