import numpy as np
from postprocessing import save_csv
import matplotlib.pyplot as plt
from ML import PolynomialRegression

np.random.seed(3)

# select case
case = 0
# case = 1
# case = 2

# ------------------------------- settings -------------------------------
if case == 0: # underfitting
    regularization, samples = 0, 8
    p = 1
elif case == 1: # ideal fitting
    regularization, samples = 0, 8
    p = 2
elif case == 2: # overfitting
    regularization, samples = 0, 8
    p = 6

y_true = lambda x : 2*x**2

# --------------------------- data generation ----------------------------
x_train = np.random.uniform(-1, 1, samples)
noise = np.random.uniform(-1, 1, samples) * 0.2
y_train = y_true(x_train) + noise

# --------------------------------- fit ----------------------------------
model = PolynomialRegression(p, regularization)
model.fit(x_train, y_train)

# ------------------------------ prediction ------------------------------
x_pred = np.linspace(-1, 1, 100)
y_pred = model.forward(x_pred)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(x_pred, y_pred, 'k')
ax.plot(x_train, y_train, 'ro')
ax.plot(x_pred, y_true(x_pred), 'b')
ax.set_ylim(-0.5, 2.5)
plt.show()

# ----------------------- postprocessing for book ------------------------
save_csv(f'../../results/polynomial_overfitting_{case}.csv',
         x=x_pred, ypred=y_pred, y=y_true(x_pred))
save_csv(f'../../results/polynomial_overfitting_train_{case}.csv',
         x=x_train, y=y_train)