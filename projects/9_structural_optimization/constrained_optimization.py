import matplotlib.pyplot as plt
import numpy as np

from solvers.optimization import MMA

# -------------------------------------- settings -------------------------------------
# test problem
VARIABLES = 8
COEFF = np.arange(1, VARIABLES + 1) ** 2.0
COEFF = COEFF / np.sum(COEFF)  # scaled so the objective is one at the upper bound

# optimization
BUDGET = 0.3
DCONSTRAINT = np.full(VARIABLES, 1.0 / VARIABLES)
GAMMA_MIN = 0.01
ITERS = 300
LR = 0.005
ETA = 0.5  # damping exponent of the optimality criteria update
BETA = 200.0  # penalty factor
GROWTH = 1.01  # continuation factor, applied every penalty iteration
INNER_ITERS = 5  # gradient steps between two multiplier updates
TOL = 1e-9

# --------------------------------------- helper --------------------------------------
def cost(gamma):
    return np.sum(COEFF / gamma)


def dcost(gamma):
    return -COEFF / gamma**2


def constraint(gamma):
    return np.mean(gamma) - BUDGET


def penalty(gamma):
    history = [gamma]
    beta = BETA
    for _ in range(ITERS):
        gradient = dcost(gamma) + beta * max(0.0, constraint(gamma)) * DCONSTRAINT
        gamma = np.clip(gamma - LR * gradient, GAMMA_MIN, 1.0)
        beta *= GROWTH
        history.append(gamma)
    return np.array(history)


def augmented_lagrangian(gamma):
    history = [gamma]
    mu = 0.0
    for _ in range(ITERS // INNER_ITERS):
        for _ in range(INNER_ITERS):
            estimate = max(0.0, mu + BETA * constraint(gamma))
            gradient = dcost(gamma) + estimate * DCONSTRAINT
            gamma = np.clip(gamma - LR * gradient, GAMMA_MIN, 1.0)
            history.append(gamma)
        mu = max(0.0, mu + BETA * constraint(gamma))
    return np.array(history)


def optimality_criteria(gamma):
    history = [gamma]
    for _ in range(ITERS):
        sensitivity = dcost(gamma)
        mu_lower, mu_upper = TOL, 1.0 / TOL
        while mu_upper - mu_lower > TOL * mu_upper:
            mu = 0.5 * (mu_lower + mu_upper)
            ratio = -sensitivity / (mu * DCONSTRAINT)
            gamma_new = np.clip(gamma * ratio**ETA, GAMMA_MIN, 1.0)
            if constraint(gamma_new) > 0.0:
                mu_lower = mu
            else:
                mu_upper = mu
        gamma = gamma_new
        history.append(gamma)
    return np.array(history)


def mma(gamma):
    optimizer = MMA(VARIABLES)
    history = [gamma]
    for _ in range(ITERS):
        gamma_new = optimizer.step(gamma, cost(gamma), dcost(gamma),
                                   constraint(gamma), DCONSTRAINT)
        gamma = np.clip(gamma_new.ravel(), GAMMA_MIN, 1.0)
        history.append(gamma)
    return np.array(history)


# ------------------------------------ optimization -----------------------------------
gamma_exact = BUDGET * VARIABLES * np.sqrt(COEFF) / np.sum(np.sqrt(COEFF))
cost_exact = cost(gamma_exact)

gamma_start = np.full(VARIABLES, BUDGET)
designs = [method(gamma_start) for method in
           (penalty, augmented_lagrangian, optimality_criteria, mma)]

# ----------------------------------- postprocessing ----------------------------------
error = [np.array([cost(g) for g in d]) / cost_exact - 1.0 for d in designs]
violation = [np.array([max(0.0, constraint(g)) for g in d]) for d in designs]

for name, d in zip(("penalty", "augmented lagrangian", "oc", "mma"), designs):
    print(f"{name:21s} cost {cost(d[-1]):.6f} "
          f"violation {max(0.0, constraint(d[-1])):.2e}")
print(f"{'exact':21s} cost {cost_exact:.6f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for d, e, v, color in zip(designs, error, violation, ("k", "r", "b", "g")):
    axes[0].semilogy(np.abs(e) + 1e-16, color=color)
    axes[1].plot(v, color=color)
    axes[2].step(np.arange(VARIABLES), d[-1], color=color, where="mid")
axes[2].step(np.arange(VARIABLES), gamma_exact, color="gray", where="mid", ls="--")
plt.show()
