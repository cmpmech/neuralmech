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

np.random.seed(0)

# ------------------------------ regression ------------------------------
x = np.random.uniform(0, 2, 30)
y = x**2 + np.random.randn(30) * 0.2

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x, y, "ro")
    plt.show()

if args.book:
    save_csv(RESULTS_DIR / "ml_tasks_reg.csv", x=x, y=y)

# ---------------------------- classification ----------------------------
x1 = np.random.uniform(0, 2, 30)
y1 = np.random.uniform(0, 1, 30)
x2 = np.random.uniform(0.5, 3, 30)
y2 = np.random.uniform(1.5, 3, 30)

x = np.linspace(0, 3, 100)

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x1, y1, "ko")
    ax.plot(x2, y2, "ro")
    ax.plot(x, -0.2 * x + 1.5)
    plt.show()

if args.book:
    save_csv(RESULTS_DIR / "ml_tasks_clas.csv", x1=x1, y1=y1, x2=x2, y2=y2)

# ----------------------- representation learning ------------------------
x1 = np.random.uniform(0, 1, 30)
y1 = np.random.uniform(0, 1, 30)
x2 = np.random.uniform(1, 2.5, 30)
y2 = np.random.uniform(1.5, 3, 30)

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x1, y1, "ko")
    ax.plot(x2, y2, "ro")
    plt.show()

if args.book:
    save_csv(RESULTS_DIR / "ml_tasks_rep.csv", x1=x1, y1=y1, x2=x2, y2=y2)

# ------------------------- generative modeling --------------------------
x1 = np.random.uniform(0, 1, 60)
y1 = np.random.uniform(0, 1, 60)
x2 = np.random.uniform(0, 1, 20)
y2 = np.random.uniform(0, 1, 20)

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x1, y1, "ko")
    ax.plot(x2, y2, "ro")
    plt.show()

if args.book:
    save_csv(RESULTS_DIR / "ml_tasks_gen1.csv", x=x1, y=y1)
    save_csv(RESULTS_DIR / "ml_tasks_gen2.csv", x=x2, y=y2)
