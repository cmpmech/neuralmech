import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# ------------------------------- momentum -------------------------------
x = np.array([0, 0.3, 0.6, 1.3, 2.5])
y = np.array([2, 0.3, 0.8, 0, 2])

coefficients = np.polyfit(x, y, 4)
f = np.poly1d(coefficients)
dfdx = f.deriv()

x_ = np.linspace(-0.1, 2.5, 100)
y_ = f(x_)

# gradient descent
alpha = 0.01
xgd = np.zeros(16)
xgd[0] = -0.05
for i in range(len(xgd) - 1):
    xgd[i + 1] = xgd[i] - alpha * dfdx(xgd[i])

# gradient descent with momentum
alpha = 0.01
eta = 0.9
xgdm = np.zeros(16)
vm = np.zeros(16)
xgdm[0] = -0.05
for i in range(len(xgdm) - 1):
    vm[i + 1] = eta * vm[i] - alpha * dfdx(xgdm[i])
    xgdm[i + 1] = xgdm[i] + vm[i + 1]

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_, y_, "k")
    ax.plot(xgd, f(xgd), "ro")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_, y_, "k")
    ax.plot(xgdm, f(xgdm), "ro")
    plt.show()

if args.book:
    save_csv(RESULTS_DIR / "momentum.csv", x=xgd, y=f(xgd), xm=xgdm, ym=f(xgdm))

# ------------------------------- adagrad --------------------------------
x = np.array([-1, 0, 1])
y = np.array([2, 0, 2])

coefficients = np.polyfit(x, y, 2)
f = np.poly1d(coefficients)
dfdx = f.deriv()

x_ = np.linspace(-2, 2, 100)
y_ = f(x_)

# gradient descent large learning rate
alpha = 0.45
xgdl = np.zeros(15)
xgdl[0] = -1.8
for i in range(len(xgdl) - 1):
    xgdl[i + 1] = xgdl[i] - alpha * dfdx(xgdl[i])

# gradient descent small learning rate
alpha = 0.03
xgds = np.zeros(15)
xgds[0] = -1.8
for i in range(len(xgds) - 1):
    xgds[i + 1] = xgds[i] - alpha * dfdx(xgds[i])

# adagrad
alpha = 0.9
xgd = np.zeros(15)
xgd[0] = -1.8
gt = 0
epsilon = 1e-8
for i in range(len(xgd) - 1):
    gt += dfdx(xgd[i]) ** 2
    xgd[i + 1] = xgd[i] - alpha / np.sqrt(gt + epsilon) * dfdx(xgd[i])

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_, y_, "k")
    ax.plot(xgdl, f(xgdl), "ro")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_, y_, "k")
    ax.plot(xgds, f(xgds), "ro")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_, y_, "k")
    ax.plot(xgd, f(xgd), "ro")
    plt.show()

if args.book:
    save_csv(
        RESULTS_DIR / "adagrad.csv",
        xs=xgds,
        ys=f(xgds),
        xl=xgdl,
        yl=f(xgdl),
        x=xgd,
        y=f(xgd),
    )
