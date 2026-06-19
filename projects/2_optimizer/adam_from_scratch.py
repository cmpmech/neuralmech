import matplotlib.pyplot as plt
import numpy as np


class Adam:
    def __init__(self, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0.0
        self.v = 0.0
        self.t = 0

    def step(self, x, g):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ------------------------------------- 1D example ------------------------------------
def f(x):
    return (x - 2) ** 4 + (x - 2) ** 2


def grad_f(x):
    return 4 * (x - 2) ** 3 + 2 * (x - 2)


GUESS = 7.0
LR = 0.3
EPOCHS = 30

x = np.array([GUESS])
opt = Adam(lr=LR)
x_history = [x[0]]

for _ in range(EPOCHS):
    x = opt.step(x, grad_f(x))
    x_history.append(x[0])

print(f"minimum at x={x[0]:.6f}, f(x)={f(x).item():.2e}")

# ----------------------------------- postprocessing ----------------------------------
xs = np.linspace(-1, 8, 400)
fig, ax = plt.subplots()
ax.plot(xs, f(xs), "k")
for k in range(len(x_history) - 1):
    ax.plot(
        [x_history[k], x_history[k + 1]],
        [f(x_history[k]), f(x_history[k + 1])],
        "-o",
        color=plt.cm.Greys(0.2 + 0.8 * k / len(x_history)),
    )
plt.show()
