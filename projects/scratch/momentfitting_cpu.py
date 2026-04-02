"""
Voxel Moment Fitting Quadrature
================================
Steps:
  1. GL points on reference element [-1,1]^D — Lagrange Stuetzstellen.
  2. Voxel sub-grid with GL points inside each voxel.
  3. Lagrange basis pre-evaluated at all voxel IPs (CPU).
  4. alpha_func evaluated at every voxel IP        (CPU, Python callable).
  5. Moment accumulation via tensor-product loop   (CPU).
"""

import csv
import itertools
import math
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Gauss-Legendre points on [-1, 1]
# ---------------------------------------------------------------------------

def gauss_legendre_points(n: int):
    """Return (pts, weights) for n-point GL rule on [-1,1]."""
    pts, wts = np.polynomial.legendre.leggauss(n)
    return pts, wts


# ---------------------------------------------------------------------------
# 1-D Lagrange basis evaluation
# ---------------------------------------------------------------------------

def lagrange_eval(nodes: np.ndarray, eval_pts: np.ndarray) -> np.ndarray:
    """
    Evaluate all Lagrange basis polynomials defined by *nodes* at *eval_pts*.

    Returns L of shape (len(eval_pts), len(nodes)):
      L[j, i] = L_i( eval_pts[j] )
    """
    m = len(nodes)
    n = len(eval_pts)
    L = np.ones((n, m))
    for i in range(m):
        for k in range(m):
            if k != i:
                L[:, i] *= (eval_pts - nodes[k]) / (nodes[i] - nodes[k])
    return L


# ---------------------------------------------------------------------------
# Build voxel grid in [-1,1] for one axis
# ---------------------------------------------------------------------------

def build_voxel_grid_1d(n_voxels: int, n_quad: int):
    """
    Tile n_voxels sub-intervals over [-1,1], place n_quad GL points in each.

    Returns pts, weights (concatenated), step (voxel width).
    """
    step = 2.0 / n_voxels
    gl_pts, gl_wts = gauss_legendre_points(n_quad)

    pts, weights = [], []
    for v in range(n_voxels):
        lo = -1.0 + v * step
        mapped = lo + step * 0.5 * (gl_pts + 1.0)
        pts.extend(mapped.tolist())
        weights.extend(gl_wts.tolist())

    return np.array(pts), np.array(weights), step


# ---------------------------------------------------------------------------
# Core: compute moments for one element (CPU)
# ---------------------------------------------------------------------------

def compute_moments(
    nvoxels: tuple,
    polynomial_degree: int,
    alpha_func,           # callable: (rst: np.ndarray) -> float
) -> tuple:
    """
    Compute the moment-fitting weight vector for one element on [-1,1]^D.

    Parameters
    ----------
    nvoxels           : number of voxels per axis, length D
    polynomial_degree : p — Lagrange basis has 2p+1 nodes per axis (stride)
    alpha_func        : material/indicator function evaluated at reference coords

    Returns
    -------
    moments          : np.ndarray, shape (stride^D,)
    elem_pts_per_axis: list of D arrays of GL nodes (Stuetzstellen)
    """
    D      = len(nvoxels)
    stride = 2 * polynomial_degree + 1   # number of Lagrange nodes per axis

    # ---- 1. Element GL nodes (Stuetzstellen) --------------------------------
    elem_pts_per_axis = []
    for _ in range(D):
        pts, _ = gauss_legendre_points(stride)
        elem_pts_per_axis.append(pts)

    # ---- 2. Voxel integration points ----------------------------------------
    n_voxel_quad = math.ceil(stride / 2.0)

    voxel_pts_per_axis = []
    voxel_wts_per_axis = []
    steps = []
    for d in range(D):
        pts, wts, step = build_voxel_grid_1d(nvoxels[d], n_voxel_quad)
        voxel_pts_per_axis.append(pts)
        voxel_wts_per_axis.append(wts)
        steps.append(step)

    voxel_detJ = np.prod([0.5 * s for s in steps])

    # ---- 3. Lagrange evaluation: L[d][j, i] = L_i( voxel_pts[d][j] ) -------
    lag_eval = []
    for d in range(D):
        L = lagrange_eval(elem_pts_per_axis[d], voxel_pts_per_axis[d])
        lag_eval.append(L)   # shape (n_vox_d, stride)

    # ---- 4. Accumulate moments -----------------------------------------------
    moments = np.zeros(stride ** D)
    n_vox   = [len(voxel_pts_per_axis[d]) for d in range(D)]

    for ijk in itertools.product(*[range(n) for n in n_vox]):
        rst = np.array([voxel_pts_per_axis[d][ijk[d]] for d in range(D)])
        w   = np.prod([voxel_wts_per_axis[d][ijk[d]] for d in range(D)])
        alpha_w = alpha_func(rst) * w * voxel_detJ

        for flat_idx, lmn in enumerate(itertools.product(*[range(stride)] * D)):
            val = alpha_w
            for d in range(D):
                val *= lag_eval[d][ijk[d], lmn[d]]
            moments[flat_idx] += val

    return moments, elem_pts_per_axis


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def write_csv(path: str, elem_pts_per_axis, moments, D):
    axis_labels = ["r", "s", "t"][:D]
    header = [f"i_{ax}" for ax in axis_labels] + \
             [f"xi_{ax}" for ax in axis_labels] + \
             ["moment_weight"]
    stride = len(elem_pts_per_axis[0])
    rows = []
    for flat_idx, lmn in enumerate(itertools.product(*[range(stride)] * D)):
        coord = [elem_pts_per_axis[d][lmn[d]] for d in range(D)]
        rows.append(list(lmn) + coord + [moments[flat_idx]])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Alpha helpers
# ---------------------------------------------------------------------------

def alpha_uniform(_xyz):
    return 1.0

def alpha_circle(xyz):
    return 1.0 if np.dot(xyz, xyz) <= 1.0 else 0.0

def alpha_halfspace(xyz):
    return 1.0 if xyz[0] < 0.0 else 0.0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    D                 = 3
    polynomial_degree = 2
    alpha_func        = alpha_circle
    out_path          = "moment_weights.csv"

    timings = []
    Ns = list(range(1, 50, 10))
    for N in Ns:
        nvoxels = (N, N, N)

        tic = time.time()
        moments, elem_pts_per_axis = compute_moments(
            nvoxels=nvoxels,
            polynomial_degree=polynomial_degree,
            alpha_func=alpha_func,
        )
        toc = time.time()
        timings.append(toc - tic)

    fig, ax = plt.subplots()
    ax.plot(Ns, timings, 'k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(3e-3, 3e1)
    ax.grid()
    plt.show()