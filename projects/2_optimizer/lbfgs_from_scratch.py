# adapated from https://doi.org/10.33774/coe-2021-qpq2j
import matplotlib.pyplot as plt
import numpy as np


class Lbfgs:
    def __init__(self, k, size):
        self.k = k  # maximum number of vectors
        self.i = 0  # currently stored vectors
        self.s = np.zeros((size, k))
        self.y = np.zeros((size, k))

    def put(self, s, y):
        # when less then k vector pairs are stored, we increase the index of stored pairs and add the new pair
        if self.i < self.k:
            self.i += 1
        # otherwise we kick out the pair with the lowest index by rolling to the left and over-writing the last pair
        else:
            self.s = np.roll(self.s, -1, axis=1)
            self.y = np.roll(self.y, -1, axis=1)

        self.s[:, self.i - 1] = s
        self.y[:, self.i - 1] = y

    def iterate(self, q):

# ----------------------------- iteration 1 ------------------------------
        alpha = np.zeros(self.i)

        for n in range(self.i - 1, -1, -1):
            rho = 1.0 / np.dot(self.y[:, n], self.s[:, n])
            alpha[n] = rho * np.dot(self.s[:, n], q)
            q = q - alpha[n] * self.y[:, n]

        r = q

# ----------------------------- iteration 2 ------------------------------
        for n in range(0, self.i):
            rho = 1.0 / np.dot(self.y[:, n], self.s[:, n])
            beta = rho * np.dot(self.y[:, n], r)
            r = r + (alpha[n] - beta) * self.s[:, n]

        # return the negative descent direction
        return r


# ------------------------------ 1D example ------------------------------
def f(x):
    return (x - 2) ** 4 + (x - 2) ** 2


def grad_f(x):
    return 4 * (x - 2) ** 3 + 2 * (x - 2)


GUESS = 7.0
EPOCHS = 30
STORED_VECS = 5

x = np.array([GUESS])
opt = Lbfgs(k=STORED_VECS, size=1)
x_history = [x[0]]

for _ in range(EPOCHS):
    g = grad_f(x)
    d = -opt.iterate(g) if opt.i > 0 else -g

    alpha = 1.0  # step size for line search
    while f(x + alpha * d) > f(x) - 1e-4 * alpha * np.dot(g, d):
        alpha *= 0.5

    x_new = x + alpha * d
    s, y = x_new - x, grad_f(x_new) - g
    if np.dot(y, s) > 1e-10:
        opt.put(s, y)
    x = x_new
    x_history.append(x[0])

print(f"minimum at x={x[0]:.6f}, f(x)={f(x).item():.2e}")

# ---------------------------- postprocessing ----------------------------
xs = np.linspace(-1, 8, 400)
fig, ax = plt.subplots()
ax.plot(xs, f(xs), "k", lw=1.5)
for k in range(len(x_history) - 1):
    ax.plot(
        [x_history[k], x_history[k + 1]],
        [f(x_history[k]), f(x_history[k + 1])],
        "-o",
        ms=5,
        lw=1,
        color=plt.cm.Greys(0.2 + 0.8 * k / len(x_history)),
    )
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
plt.show()
