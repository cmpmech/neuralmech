import numpy as np
import matplotlib.pyplot as plt
from postprocessing import save_csv
from ML import LogisticRegression

np.random.seed(1)

# --------------------------- data generation ----------------------------
x_train = np.random.randn(40, 2)
y_train = (2 * x_train[:, 0] + x_train[:, 1] > 1 + 0.1 * np.random.randn(40)).astype(np.float64)

x_val = np.random.randn(20, 2)
y_val = (2 * x_val[:, 0] + x_val[:, 1] > 1 + 0.1 * np.random.randn(20)).astype(np.float64)

# ----------------------------- optimization -----------------------------
model = LogisticRegression()
lr = 1e0
epochs = 200

train_cost, val_cost = model.train(epochs, lr, x_train, y_train,
                                   x_val, y_val)

print(f'weights={model.weights}, bias={model.bias}')

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
ax.plot(val_cost, 'r')
plt.show()

x1 = np.linspace(-3, 3, 2)
x2 = -(model.weights[0] * x1 + model.bias) / model.weights[1] # when = 0.5

fig, ax = plt.subplots()
ax.plot(x_train[y_train == 0, 0], x_train[y_train == 0, 1], 'bs')
ax.plot(x_train[y_train == 1, 0], x_train[y_train == 1, 1], 'bo')
ax.plot(x_val[y_val == 0, 0], x_val[y_val == 0, 1], 'rs')
ax.plot(x_val[y_val == 1, 0], x_val[y_val == 1, 1], 'ro')
ax.plot(x1, x2, 'k')
plt.show()

# ------------------------- book postprocessing --------------------------
save_csv(f'../../results/logistic_regression_train0.csv',
         x1=x_train[y_train==0,0], x2=x_train[y_train==0,1])
save_csv(f'../../results/logistic_regression_train1.csv',
         x1=x_train[y_train==1,0], x2=x_train[y_train==1,1])