import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(1)

# select case
case = 0
# case = 1
# case = 2

# --------------------------- data generation ----------------------------
x_train = np.random.randn(16)
y_train = 2 * x_train + 3 + np.random.randn(16)

x_train[0] = -2.5
y_train[0] += 10

x_val = np.random.randn(4)
y_val = 2 * x_val + 3 + np.random.randn(4)

# -------------------------- linear regression ---------------------------
if case == 0: # MSE
    model = LinearRegression()
elif case == 1: # MAE
    model = QuantileRegressor(quantile=0.5, alpha=0)
elif case == 2: # Huber
    model = HuberRegressor()
model.fit(x_train.reshape(-1, 1), y_train)

print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# ---------------------------- postprocessing ----------------------------
x_test = np.linspace(-3,3,2)
y_test_pred = model.predict(x_test.reshape(-1, 1))
fig, ax = plt.subplots()
ax.plot(x_test, y_test_pred, 'k')
ax.plot(x_train, y_train, 'ko')
ax.plot(x_val, y_val, 'ro')
plt.show()

# ------------------------- book postprocessing --------------------------
df = pd.DataFrame({'x': x_train,
                   'y': y_train})
df.to_csv(f'../../results/linear_regression_outlier_train.csv', sep=' ', index=False)
df = pd.DataFrame({'x': x_val,
                   'y': y_val})
df.to_csv(f'../../results/linear_regression_outlier_val.csv', sep=' ', index=False)