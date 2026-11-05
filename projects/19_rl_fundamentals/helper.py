import numpy as np
import torch

# physics
GRAVITY = 9.8  # m/s^2
CART_MASS = 1.0  # kg
POLE_MASS = 0.1  # kg
POLE_LENGTH = 0.5  # m, half the pole length
FORCE = 10.0  # N, magnitude of a push
DT = 0.02  # s, explicit euler time step
MAX_STEPS = 500  # episode cap, also the maximum return
X_LIMIT = 2.4  # m
THETA_LIMIT = 12 * np.pi / 180  # rad


def reset_cartpole(rng: np.random.Generator) -> np.ndarray:
    """draw a start state (x, x_dot, theta, theta_dot) close to upright rest."""
    return rng.uniform(-0.05, 0.05, 4)


def step_cartpole(state: np.ndarray, action: int) -> tuple[np.ndarray, bool]:
    """advance the cart-pole by one explicit euler step; returns (state, fallen).

    Action 1 pushes right, action 0 pushes left. The equations of motion are those of
    Barto, Sutton and Anderson (1983) with a massless rod carrying the pole mass.
    """
    x, x_dot, theta, theta_dot = state
    force = FORCE if action == 1 else -FORCE
    total_mass = CART_MASS + POLE_MASS
    sin, cos = np.sin(theta), np.cos(theta)
    temp = (force + POLE_MASS * POLE_LENGTH * theta_dot**2 * sin) / total_mass
    theta_acc = (GRAVITY * sin - cos * temp) / (
        POLE_LENGTH * (4 / 3 - POLE_MASS * cos**2 / total_mass)
    )
    x_acc = temp - POLE_MASS * POLE_LENGTH * theta_acc * cos / total_mass
    state = state + DT * np.array([x_dot, x_acc, theta_dot, theta_acc])
    fallen = abs(state[0]) > X_LIMIT or abs(state[2]) > THETA_LIMIT
    return state, bool(fallen)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """return G_tau = r_tau + gamma * G_(tau+1) of every step, summed backward."""
    returns = np.zeros(len(rewards))
    G = 0.0
    for tau in reversed(range(len(rewards))):
        G = rewards[tau] + gamma * G
        returns[tau] = G
    return returns


def phase_plane(resolution: int) -> tuple[torch.Tensor, list[float]]:
    """states on a (theta, theta_dot) grid with the cart at rest in the center.

    Returns the states as a (resolution**2, 4) tensor, theta varying slowest, and the
    imshow extent in degrees and rad/s.
    """
    theta = torch.linspace(-THETA_LIMIT, THETA_LIMIT, resolution)
    theta_dot = torch.linspace(-2, 2, resolution)
    theta, theta_dot = torch.meshgrid(theta, theta_dot, indexing="ij")
    zeros = torch.zeros(resolution**2)
    states = torch.stack([zeros, zeros, theta.flatten(), theta_dot.flatten()], dim=1)
    extent = [-np.degrees(THETA_LIMIT), np.degrees(THETA_LIMIT), -2, 2]
    return states, extent
