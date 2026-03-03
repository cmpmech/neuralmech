import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)

# --------------------------- data generation ----------------------------
x_train = np.random.randn(16)
y_train = 2 * x_train + 3 + np.random.randn(16)

x_val = np.random.randn(4)
y_val = 2 * x_val + 3 + np.random.randn(4)

# ----------------------- linear regression class ------------------------
class LinearRegression:
    def __init__(self):
        self.weight = 0
        self.bias = 0

    def forward(self, x):
        return self.weight * x + self.bias

    def cost_fun(self, x, y):
        y_pred = self.forward(x)
        return np.mean((y - y_pred)**2)

    def cost_fun_grad(self, x, y):
        y_pred = self.forward(x)
        res = y_pred - y
        return 2 * np.mean(x * res), 2 * np.mean(res)

    def train(self, epochs, lr, x_train, y_train, x_val, y_val):
        train_cost = [0] * epochs
        val_cost = [0] * epochs
        for epoch in range(epochs):
            train_cost[epoch] = self.cost_fun(x_train, y_train)
            val_cost[epoch] = self.cost_fun(x_val, y_val)
            grad = self.cost_fun_grad(x_train, y_train)
            self.weight -= lr * grad[0]
            self.bias -= lr * grad[1]
        return train_cost, val_cost

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
df = pd.DataFrame({'x': x_train,
                   'y': y_train})
df.to_csv(f'../../results/linear_regression_train.csv', sep=' ', index=False)
df = pd.DataFrame({'x': x_val,
                   'y': y_val})
df.to_csv(f'../../results/linear_regression_val.csv', sep=' ', index=False)