import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(2)

# -------------------------------------- settings -------------------------------------
# select polynomial degree
# P = 1
P = 3

# ------------------------------------ create data ------------------------------------
x_train = rng.standard_normal((40, 2))
y_train = (
    2 * x_train[:, 0] + x_train[:, 1] > 1 + 0.1 * rng.standard_normal(40)
).astype(np.float64)

x_val = rng.standard_normal((20, 2))
y_val = (2 * x_val[:, 0] + x_val[:, 1] > 1 + 0.1 * rng.standard_normal(20)).astype(
    np.float64
)

# -------------------------------------- fitting --------------------------------------
# transform to polynomial features, then logistic regression
model = make_pipeline(PolynomialFeatures(degree=P), LogisticRegression())
model.fit(x_train, y_train)

# ----------------------------------- postprocessing ----------------------------------
x1, x2 = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
grid = np.column_stack([x1.ravel(), x2.ravel()])
z = model.predict(grid).reshape(x1.shape)

fig, ax = plt.subplots()
ax.contourf(x1, x2, z, cmap="cividis")
ax.contour(x1, x2, z, levels=[0.5], colors="k")
ax.plot(x_train[y_train == 0, 0], x_train[y_train == 0, 1], "bs")
ax.plot(x_train[y_train == 1, 0], x_train[y_train == 1, 1], "bo")
ax.plot(x_val[y_val == 0, 0], x_val[y_val == 0, 1], "rs")
ax.plot(x_val[y_val == 1, 0], x_val[y_val == 1, 1], "ro")
plt.show()
