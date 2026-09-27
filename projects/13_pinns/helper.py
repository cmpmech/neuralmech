import math

import numpy as np
import torch
from tqdm import tqdm

from DL import differentiate


# -------------------------------------- sampling -------------------------------------
def grid(lower, upper, resolution):
    """cell midpoints of a box in D dimensions and their midpoint-rule weights.

    The points double as collocation points and as quadrature points; returns
    x (N, D) with gradient tracking and w (N, 1).
    """
    resolution = (
        resolution if isinstance(resolution, list) else [resolution] * len(lower)
    )
    axes = []
    for a, b, n in zip(lower, upper, resolution):
        h = (b - a) / n
        axes.append(torch.linspace(a + h / 2, b - h / 2, n))
    x = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
        -1, len(lower)
    )
    volume = math.prod((b - a) / n for a, b, n in zip(lower, upper, resolution))
    w = torch.full((len(x), 1), volume)
    return x.requires_grad_(), w


def sample(lower, upper, n, kind="uniform"):
    """n collocation points in a box: uniform midpoints, random, or scrambled sobol.

    The weights volume / n turn any of them into a (quasi) Monte Carlo quadrature.
    """
    if kind == "uniform":
        return grid(lower, upper, round(n ** (1 / len(lower))))
    if kind == "random":
        unit = torch.rand(n, len(lower))
    else:
        unit = torch.quasirandom.SobolEngine(len(lower), scramble=True, seed=0).draw(n)
    lower, upper = torch.tensor(lower), torch.tensor(upper)
    x = lower + (upper - lower) * unit
    w = torch.full((n, 1), torch.prod(upper - lower).item() / n)
    return x.requires_grad_(), w


def graded_quadrature(dimension, levels, order):
    """Gauss-Legendre points on cells of [0, 1]^D graded geometrically towards the origin.

    Like a mesh refined towards the corner: starting from 2^D cells, each level splits the
    cell at the origin into 2^D children; returns x (N, D) and w (N, 1).
    """
    points, weights = np.polynomial.legendre.leggauss(order)
    axes = [torch.tensor((points + 1) / 2, dtype=torch.float32)] * dimension
    reference_x = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    reference_x = reference_x.reshape(-1, dimension)
    reference_w = torch.tensor(weights / 2, dtype=torch.float32)
    reference_w = math.prod(torch.meshgrid(*[reference_w] * dimension, indexing="ij"))
    reference_w = reference_w.reshape(-1, 1)

    corners = torch.cartesian_prod(*[torch.tensor([0.0, 1.0])] * dimension)
    corners = corners.reshape(-1, dimension)[1:]
    x, w = [], []
    for level in range(levels + 1):
        h = 0.5 ** (level + 1)
        cells = corners * h if level < levels else torch.zeros(1, dimension)
        h = h if level < levels else 2 * h
        x.append((cells[:, None, :] + h * reference_x[None]).reshape(-1, dimension))
        w.append((h**dimension * reference_w).repeat(len(cells), 1))
    return torch.cat(x), torch.cat(w)


def face(lower, upper, resolution, axis, side):
    """midpoints on one face of a box, their weights and the outward box normal.

    Args:
        axis, side: the face lies at `lower[axis]` (side 0) or `upper[axis]` (side 1).
    """
    others = [i for i in range(len(lower)) if i != axis]
    if others:
        resolution = (
            resolution if isinstance(resolution, list) else [resolution] * len(lower)
        )
        x, w = grid(
            [lower[i] for i in others],
            [upper[i] for i in others],
            [resolution[i] for i in others],
        )
        x = x.detach()
    else:
        x, w = torch.zeros(1, 0), torch.ones(1, 1)
    value = torch.full((len(x), 1), float(upper[axis] if side else lower[axis]))
    x = torch.cat([x[:, :axis], value, x[:, axis:]], dim=1)
    normal = torch.zeros(len(x), len(lower))
    normal[:, axis] = 1.0 if side else -1.0
    return x.requires_grad_(), w, normal


# -------------------------------------- geometry -------------------------------------
def in_holes(x, holes):
    """mask of the points (..., 2) inside a hole of the perforated plate [0, 1]^2.

    fifth of the cell size, so grids with multiples of 5 * holes cells per side conform.
    fifth of the cell size, so grids with 5 * holes cells per side (or multiples) conform.
    """
    cell = (x * holes) % 1
    return ((cell > 0.4) & (cell < 0.6)).all(-1)


def plate_points(holes, resolution):
    """midpoint quadrature of the perforated plate and of its loaded right edge.

    Returns x, w in the plate and x, w on the right edge.
    """
    x, w = grid([0.0, 0.0], [1.0, 1.0], resolution)
    outside = ~in_holes(x.detach(), holes)
    x_right, w_right, _ = face([0.0, 0.0], [1.0, 1.0], resolution, 0, 1)
    return x[outside].detach().requires_grad_(), w[outside], x_right, w_right


def hole_faces(holes, resolution):
    """midpoints, weights and outward plate normals on the edges of all holes."""
    size = 0.2 / holes
    faces = []
    for i in range(holes):
        for j in range(holes):
            lower = [(i + 0.4) / holes, (j + 0.4) / holes]
            upper = [lower[0] + size, lower[1] + size]
            for axis in [0, 1]:
                for side in [0, 1]:
                    x, w, n = face(lower, upper, round(resolution * size), axis, side)
                    faces.append((x, w, -n))
    return faces


# ------------------------------------- elasticity ------------------------------------
def plane_stress(E, nu):
    """plane stress material matrix in Voigt notation [xx, yy, xy]."""
    return (
        E / (1 - nu**2) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    )


def strain(u, x):
    """small strain [exx, eyy, gamma_xy] of the displacement u (N, 2)."""
    dux = differentiate(u[:, 0:1], x)
    duy = differentiate(u[:, 1:2], x)
    return torch.stack([dux[:, 0], duy[:, 1], dux[:, 1] + duy[:, 0]], dim=1)


def traction(sigma, n):
    """traction of the Voigt stress sigma (N, 3) on a surface with normal n (N, 2)."""
    tx = sigma[:, 0] * n[:, 0] + sigma[:, 2] * n[:, 1]
    ty = sigma[:, 2] * n[:, 0] + sigma[:, 1] * n[:, 1]
    return torch.stack([tx, ty], dim=1)


def energy_error(potential, compliance):
    """relative energy error of an admissible displacement from its potential energy.

    The exact solution minimizes the potential at -compliance / 2, and for any
    admissible u, Pi(u) - Pi(u_exact) is half the squared energy norm of the error.
    """
    return math.sqrt(max(potential + compliance / 2, 0.0) / (compliance / 2))


def plate_potential(u_hat, C, t, holes, resolution, device):
    """potential energy of the loaded perforated plate, integrated on a midpoint grid.

    Evaluated in chunks on a grid finer than the training points, as the network may
    fit the quadrature it was trained on.
    """
    x, w, x_right, w_right = plate_points(holes, resolution)
    C, t = C.to(device), t.to(device)
    potential = -torch.sum(w_right.to(device) * (u_hat(x_right.to(device)) @ t.T))
    for x_chunk, w_chunk in zip(x.split(2**16), w.split(2**16)):
        x_chunk = x_chunk.detach().to(device).requires_grad_()
        epsilon = strain(u_hat(x_chunk), x_chunk)
        density = 0.5 * torch.sum(epsilon * (epsilon @ C), 1, keepdim=True)
        potential = potential + torch.sum(w_chunk.to(device) * density)
    return potential.item()


# ------------------------------------ corner plots -----------------------------------
def corner_plot_points(dimension, resolution):
    """numpy points on which the corner singularity is plotted in 1, 2 or 3 dimensions.

    A logarithmic line in 1D, a midpoint grid in 2D, and in 3D midpoint grids on the
    three faces through the singular corner, face x_i = 0 after face x_(i-1) = 0.
    """
    if dimension == 1:
        return np.logspace(-4, 0, resolution)[:, None]
    axis = (np.arange(resolution) + 0.5) / resolution
    a, b = np.meshgrid(axis, axis, indexing="ij")
    if dimension == 2:
        return np.stack([a.ravel(), b.ravel()], axis=1)
    zeros = np.zeros(a.size)
    return np.concatenate(
        [
            np.stack([zeros, a.ravel(), b.ravel()], axis=1),
            np.stack([a.ravel(), zeros, b.ravel()], axis=1),
            np.stack([a.ravel(), b.ravel(), zeros], axis=1),
        ]
    )


# ---------------------------------------- costs --------------------------------------
def pinn_cost(residuals, weights=None):
    """weighted sum of the mean squared residuals (domain, boundary, or measurement).

    A weight is a scalar or a tensor with one entry per point.
    """
    weights = weights or [1.0] * len(residuals)
    return sum(torch.mean(weight * r**2) for r, weight in zip(residuals, weights))


def energy_cost(terms):
    """total potential energy, summing sum(w * density(x)) over the (x, w, density) terms."""
    return sum(torch.sum(w * density(x)) for x, w, density in terms)


def weak_cost(terms, test_functions, normalize=False):
    """mean squared weak residual over all test functions.

    Each term (x, w, integrand) contributes sum(w * integrand(x, v, dv)) to the
    residuals, where `test_functions(x)` returns all K test functions v (N, K) and dv
    (N, K, D) holds their gradients. With `normalize`, the residuals are divided by the
    H1 norm of their test function on the first term's points, which bounds the
    maximization over adversarial test functions.
    """
    r = 0
    for x, w, integrand in terms:
        v = test_functions(x)
        dv = torch.stack(
            [differentiate(v[:, k : k + 1], x) for k in range(v.shape[1])], 1
        )
        r = r + torch.sum(w * integrand(x, v, dv), dim=0)
        if normalize and x is terms[0][0]:
            norm = torch.sum(w * (v**2 + torch.sum(dv**2, dim=2)), dim=0)
    if normalize:
        return torch.mean(r**2 / norm)
    return torch.mean(r**2)


# --------------------------------------- training ------------------------------------
def train(
    cost_fun,
    params,
    epochs,
    lr,
    optimizer="adam",
    decay=1.0,
    ascent_params=None,
    ascent_lr=None,
    callback=None,
):
    """minimize `cost_fun()` over `params` and maximize it over `ascent_params`.

    The min-max for adversarial test functions or weights flips the sign of the
    ascent gradients before a shared Adam step. Both learning rates decay by the
    factor `decay` per epoch, and `callback(epoch)` runs before each step (e.g. to
    resample points); returns the cost history.
    """
    params = list(params)
    ascent_params = list(ascent_params or [])
    if ascent_params and optimizer != "adam":
        raise ValueError("gradient ascent requires adam")
    if optimizer == "adam":
        groups = [{"params": params, "lr": lr}]
        if ascent_params:
            groups.append({"params": ascent_params, "lr": ascent_lr or lr})
        optimizer = torch.optim.Adam(groups)
    else:
        optimizer = torch.optim.LBFGS(params, lr, line_search_fn="strong_wolfe")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay)

    def closure():
        optimizer.zero_grad()
        cost = cost_fun()
        cost.backward()
        for param in ascent_params:
            param.grad = -param.grad
        return cost

    history = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        if callback is not None:
            callback(epoch)
        cost = optimizer.step(closure)
        scheduler.step()
        history.append(cost.item())
        if epoch % 10 == 0:
            pbar.set_postfix({"cost": f"{history[-1]:.2e}"})
    return history


def least_squares(residuals_fun, params):
    """one Gauss-Newton step on the concatenated residuals with respect to `params`.

    Exact when the residuals are linear in `params`, e.g. a linear PDE with only the
    output layer free (extreme learning machine). The Jacobian is assembled row by
    row with autograd, and the least-squares problem is solved in double precision.
    """
    r = torch.cat([residual.flatten() for residual in residuals_fun()])
    rows = []
    for ri in r:
        grads = torch.autograd.grad(ri, params, retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in grads]))
    A = torch.stack(rows).double()
    delta = torch.linalg.lstsq(A, -r.detach().double().unsqueeze(1)).solution
    with torch.no_grad():
        offset = 0
        for param in params:
            param += delta[offset : offset + param.numel()].reshape(param.shape).float()
            offset += param.numel()
