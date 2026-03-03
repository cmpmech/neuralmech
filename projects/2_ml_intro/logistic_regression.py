import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1)

# --------------------------- data generation ----------------------------
x_train = np.random.randn(40, 2)
y_train = (2 * x_train[:, 0] + x_train[:, 1] > 1 + 0.1 * np.random.randn(40)).astype(np.float64)

x_val = np.random.randn(20, 2)
y_val = (2 * x_val[:, 0] + x_val[:, 1] > 1 + 0.1 * np.random.randn(20)).astype(np.float64)

# ---------------------- logistic regression class -----------------------
class LogisticRegression:
    def __init__(self):
        self.weights = np.zeros(2)
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        z = x@self.weights + self.bias
        return self.sigmoid(z)

    def cost_fun(self, x, y):
        y_pred = self.forward(x)
        # binary cross-entropy
        eps = 1e-7  # for numerical stability TODO check
        return -np.mean(y * np.log(y_pred + eps) +
                       (1 - y) * np.log(1 - y_pred + eps))

    def cost_fun_grad(self, x, y):
        y_pred = self.forward(x)
        res = y_pred - y
        return np.mean(x.T * res, 1), np.mean(res)

    def train(self, epochs, lr, x_train, y_train, x_val, y_val):
        train_cost = [0] * epochs
        val_cost = [0] * epochs
        for epoch in range(epochs):
            train_cost[epoch] = self.cost_fun(x_train, y_train)
            val_cost[epoch] = self.cost_fun(x_val, y_val)
            grad_w, grad_b = self.cost_fun_grad(x_train, y_train)
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b
        return train_cost, val_cost

    def predict(self, x):
        return (self.forward(x) > 0.5).astype(float)

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
df = pd.DataFrame({'x1': x_train[y_train==0,0],
                   'x2': x_train[y_train==0,1]})
df.to_csv(f'../../results/logistic_regression_train0.csv', sep=' ', index=False)
df = pd.DataFrame({'x1': x_train[y_train==1,0],
                   'x2': x_train[y_train==1,1]})
df.to_csv(f'../../results/logistic_regression_train1.csv', sep=' ', index=False)