import numpy as np
from postprocessing import save_csv
import matplotlib.pyplot as plt
from ML import LinearRegression

np.random.seed(1)

# --------------------------- data generation ----------------------------
x_train = np.random.randn(16)
y_train = 2 * x_train + 3 + np.random.randn(16)

x_val = np.random.randn(4)
y_val = 2 * x_val + 3 + np.random.randn(4)

# ----------------------------- optimization -----------------------------
model = LinearRegression()
lr = 1e-2
epochs = 200

train_cost, val_cost = model.train(epochs, lr, x_train, y_train,
                                   x_val, y_val)

print(f'weight w={model.weight} & bias b={model.bias}')
# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
ax.plot(val_cost, 'r')
plt.show()

x_test = np.linspace(-3,3,2)
y_test_pred = model.forward(x_test)
fig, ax = plt.subplots()
ax.plot(x_test, y_test_pred, 'k')
ax.plot(x_train, y_train, 'ko')
ax.plot(x_val, y_val, 'ro')
plt.show()

# --------------------------- normal equations ---------------------------
X = np.vstack((x_train, np.ones_like(x_train))).T
y = y_train

weight, bias = np.linalg.inv(X.T@X)@X.T@y
print(f'weight w={weight} & bias b={bias}')

# ------------------------- book postprocessing --------------------------
save_csv(f'../../results/linear_regression_train.csv', x=x_train, y=y_train)
save_csv(f'../../results/linear_regression_val.csv', x=x_val, y=y_val)