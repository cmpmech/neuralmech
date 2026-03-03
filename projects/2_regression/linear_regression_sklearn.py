import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(1)

# select polynomial degree
p = 1
# p = 10

# --------------------------- data generation ----------------------------
x_train = np.random.randn(16)
y_train = 2 * x_train + 3 + np.random.randn(16)

x_val = np.random.randn(4)
y_val = 2 * x_val + 3 + np.random.randn(4)

# ----------------------------- fitting ----------------------------------
# step 1: transform to polynomial features
# step 2: perform linear regression
model = make_pipeline(PolynomialFeatures(degree=p),
                      LinearRegression())
model.fit(x_train.reshape(-1, 1), y_train)

# ---------------------------- postprocessing ----------------------------
x_test = np.linspace(-3, 3, 100)
y_test_pred = model.predict(x_test.reshape(-1, 1))

fig, ax = plt.subplots()
ax.plot(x_test, y_test_pred, 'k')
ax.plot(x_train, y_train, 'ko')
ax.plot(x_val, y_val, 'ro')
ax.set_ylim(-3, 10)
plt.show()