import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(2)

# select polynomial degree
P = 1
# P = 10

# --------------------------- data generation ----------------------------
x_train = rng.standard_normal(16)
y_train = 2 * x_train + 3 + rng.standard_normal(16)

x_val = rng.standard_normal(4)
y_val = 2 * x_val + 3 + rng.standard_normal(4)

# ----------------------------- fitting ----------------------------------
# step 1: transform to polynomial features
# step 2: perform linear regression
model = make_pipeline(PolynomialFeatures(degree=P), LinearRegression())
model.fit(x_train.reshape(-1, 1), y_train)

# ---------------------------- postprocessing ----------------------------
x_test = np.linspace(-3, 3, 100)
y_test_pred = model.predict(x_test.reshape(-1, 1))

fig, ax = plt.subplots()
ax.plot(x_test, y_test_pred, "k")
ax.plot(x_train, y_train, "ko")
ax.plot(x_val, y_val, "ro")
ax.set_ylim(-3, 10)
plt.show()
