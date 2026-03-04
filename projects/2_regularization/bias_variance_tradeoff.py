import numpy as np
from postprocessing import save_csv
import matplotlib.pyplot as plt
from ML import PolynomialRegression

np.random.seed(1)

# select case
case = 0
# case = 1
# case = 2

# ------------------------------- settings -------------------------------
if case == 0: # number of data points scaling
    regularization, p = 0, 5
    samples = 8
    # samples = 12
    # samples = 16
elif case == 1: # capacity
    regularization, samples = 0, 16
    p = 2
    # p = 5
    # p = 7
elif case == 2: # regularization
    p, samples = 7, 16
    regularization = 0
    # regularization = 1e-5
    # regularization = 1e-1

num_fits = 1000
y_true = lambda x : np.cos(np.pi * x)

y_preds = []
for i in range(num_fits):
# --------------------------- data generation ----------------------------
    x_train = np.random.uniform(-1, 1, samples)
    x_train[0] = -1
    x_train[1] = 1
    noise = np.random.uniform(-1, 1, samples) * 0.1
    y_train = y_true(x_train) + noise

# --------------------------------- fit ----------------------------------
    model = PolynomialRegression(p, regularization)
    model.fit(x_train, y_train)

# ------------------------------ prediction ------------------------------
    x_pred = np.linspace(-1, 1, 100)
    y_pred = model.forward(x_pred)
    y_preds.append(y_pred)

y_preds = np.vstack(y_preds)
y_pred_mean = np.mean(y_preds, axis=0)
y_pred_std = np.std(y_preds, axis=0)

variance = y_pred_std**2
bias = y_pred_mean - y_true(x_pred)

# ---------------------------- postprocessing ----------------------------
print(f'bias: {np.mean(np.abs(bias)):.2e}, variance: {np.mean(variance):.2e}')

# variance
fig, ax = plt.subplots()
for i in range(num_fits):
    ax.plot(x_pred, y_preds[i], 'k', alpha=0.01)
ax.plot(x_pred, y_pred_mean, 'r')
ax.fill_between(x_pred, # 95 % confidence interval
                y_pred_mean - 2 * y_pred_std,
                y_pred_mean + 2 * y_pred_std,
                color='r', alpha=0.2)
ax.plot(x_pred, y_pred_mean - 2 * y_pred_std, 'r')
ax.plot(x_pred, y_pred_mean + 2 * y_pred_std, 'r')
ax.set_ylim(-2, 2)
plt.show()

# bias
fig, ax = plt.subplots()
for i in range(num_fits):
    ax.plot(x_pred, y_preds[i], 'k', alpha=0.01)
ax.plot(x_pred, y_pred_mean, 'b')
ax.plot(x_pred, y_true(x_pred), 'k')
ax.fill_between(x_pred, # bias
                y_true(x_pred),
                y_pred_mean,
                color='b', alpha=0.2)

ax.set_ylim(-2, 2)
plt.show()

# ------------------------- book postprocessing --------------------------
data = {'x' : x_pred, 'mean' : y_pred_mean, 'std' : y_pred_std,
        'true' : y_true(x_pred)}

for i in range(100):
    data[f'y_pred_{i}'] = y_preds[i]
if case == 0:
    save_csv(f'../../results/polynomial_regression_{case}_{samples}.csv', **data)
elif case == 1:
    save_csv(f'../../results/polynomial_regression_{case}_{p}.csv', **data)
elif case == 2:
    save_csv(f'../../results/polynomial_regression_{case}_{regularization}.csv', **data)