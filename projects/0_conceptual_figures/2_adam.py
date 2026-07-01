import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- momentum -------------------------------------
# create function from points
x_data = np.array([0, 0.3, 0.6, 1.3, 2.5])
y_data = np.array([2, 0.3, 0.8, 0, 2])
coefficients = np.polyfit(x_data, y_data, 4)
f = np.poly1d(coefficients)
dfdx = f.deriv()

# sample function
x = np.linspace(-0.1, 2.5, 100)
y = f(x)

# gradient descent
alpha = 0.01
x_gd = np.zeros(16)
x_gd[0] = -0.05
for i in range(len(x_gd) - 1):
    x_gd[i + 1] = x_gd[i] - alpha * dfdx(x_gd[i])

# gradient descent with momentum
alpha = 0.01
eta = 0.9
x_gdm = np.zeros(16)
vm = np.zeros(16)
x_gdm[0] = -0.05
for i in range(len(x_gdm) - 1):
    vm[i + 1] = eta * vm[i] - alpha * dfdx(x_gdm[i])
    x_gdm[i + 1] = x_gdm[i] + vm[i + 1]

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x, y, "k")
    ax.plot(x_gd, f(x_gd), "ro")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y, "k")
    ax.plot(x_gdm, f(x_gdm), "ro")
    plt.show()

if args.book:
    save_csv(CSV_DIR / "momentum.csv", x=x_gd, y=f(x_gd), xm=x_gdm, ym=f(x_gdm))

# -------------------------------------- adagrad --------------------------------------
x_data = np.array([-1, 0, 1])
y_data = np.array([2, 0, 2])

coefficients = np.polyfit(x_data, y_data, 2)
f = np.poly1d(coefficients)
dfdx = f.deriv()

x = np.linspace(-2, 2, 100)
y = f(x)

# gradient descent large learning rate
alpha = 0.45
x_gdl = np.zeros(15)
x_gdl[0] = -1.8
for i in range(len(x_gdl) - 1):
    x_gdl[i + 1] = x_gdl[i] - alpha * dfdx(x_gdl[i])

# gradient descent small learning rate
alpha = 0.03
x_gds = np.zeros(15)
x_gds[0] = -1.8
for i in range(len(x_gds) - 1):
    x_gds[i + 1] = x_gds[i] - alpha * dfdx(x_gds[i])

# adagrad
alpha = 0.9
x_gd = np.zeros(15)
x_gd[0] = -1.8
gt = 0
epsilon = 1e-8
for i in range(len(x_gd) - 1):
    gt += dfdx(x_gd[i]) ** 2
    x_gd[i + 1] = x_gd[i] - alpha / np.sqrt(gt + epsilon) * dfdx(x_gd[i])

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x, y, "k")
    ax.plot(x_gdl, f(x_gdl), "ro")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y, "k")
    ax.plot(x_gds, f(x_gds), "ro")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y, "k")
    ax.plot(x_gd, f(x_gd), "ro")
    plt.show()

if args.book:
    save_csv(
        CSV_DIR / "adagrad.csv",
        xs=x_gds,
        ys=f(x_gds),
        xl=x_gdl,
        yl=f(x_gdl),
        x=x_gd,
        y=f(x_gd),
    )
