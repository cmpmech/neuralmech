import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(1)

# select polynomial degree
# p = 1
p = 3

# --------------------------- data generation ----------------------------
x_train = np.random.randn(40, 2)
y_train = (2 * x_train[:, 0] + x_train[:, 1] > 1 + 0.1 * np.random.randn(40)).astype(np.float64)

x_val = np.random.randn(20, 2)
y_val = (2 * x_val[:, 0] + x_val[:, 1] > 1 + 0.1 * np.random.randn(20)).astype(np.float64)

# ----------------------------- fitting ----------------------------------
# step 1: transform to polynomial features
# step 2: perform linear regression
model = make_pipeline(PolynomialFeatures(degree=p),
                      LogisticRegression())
model.fit(x_train, y_train)

# ---------------------------- postprocessing ----------------------------
x1, x2 = np.meshgrid(np.linspace(-3, 3, 200),
                     np.linspace(-3, 3, 200))
grid = np.column_stack([x1.ravel(), x2.ravel()])
z = model.predict(grid).reshape(x1.shape)

fig, ax = plt.subplots()
ax.contourf(x1, x2, z, cmap='plasma')
ax.contour(x1, x2, z, levels=[0.5], colors='k')
ax.plot(x_train[y_train == 0, 0], x_train[y_train == 0, 1], 'bs')
ax.plot(x_train[y_train == 1, 0], x_train[y_train == 1, 1], 'bo')
ax.plot(x_val[y_val == 0, 0], x_val[y_val == 0, 1], 'rs')
ax.plot(x_val[y_val == 1, 0], x_val[y_val == 1, 1], 'ro')
plt.show()