import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from helper import (
    build_cut_graph,
    compute_sdf,
    geometric_kt,
    lognormal_field,
    net_section_area,
    run_walkers,
    save_vtk_voxels,
    solve_min_cut,
    surface_curvatures,
    unzip_response,
    walker_amplification,
)
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RESOLUTION = 64
NU = 0.3
E = 1.0
STRENGTH = 1.0
TOUGHNESS = 0.3
LATERAL = 0.3

# walker probe (iterations; jumps advance ~2.7 lattice steps per iteration at
# 64^3, so the measured walker time matches the pre-acceleration 6144/2048 run)
WALKERS = 2_000_000
STEPS = 2176
BURNIN = 640

# material fluctuations (0 = deterministic benchmark)
STRENGTH_SIGMA = 0.0
CORR_VOXELS = 4.0

# -------------------------------------- load data ------------------------------------
data = np.load(DATA_DIR / f"void_cube_{RESOLUTION}.npz")
solid = data["indicator"] > 0
h = float(data["Lx"]) / RESOLUTION

rng = np.random.default_rng(2)
strength = STRENGTH * np.ones(solid.shape)
if STRENGTH_SIGMA > 0:
    strength *= lognormal_field(solid.shape, CORR_VOXELS, STRENGTH_SIGMA, rng)

# ------------------------------ geometric amplification ------------------------------
tic = time.perf_counter()
sdf = compute_sdf(solid, h)
kappas, normal = surface_curvatures(solid, sdf, h)
kt = geometric_kt(sdf, kappas, normal, load_axis=2, nu=NU)
area = net_section_area(solid, h)

# amplification over gross nominal stress: concentration referenced to the net
# section (the standard finite-width convention; the walkers confirm the plane
# enhancement A_gross/A_net independently via flux conservation)
net_factor = (1.0 / area)[None, None, :]
amp = np.maximum(np.maximum(kt * net_factor, net_factor), 0.05)
toc_geometry = time.perf_counter() - tic

# ----------------------------------- walker probe ------------------------------------
tic = time.perf_counter()
walk = run_walkers(solid, h, WALKERS, STEPS, BURNIN, device, seed=0)
shape_s = np.where(kappas[..., 0] > 0, np.clip(kappas[..., 1] / np.maximum(kappas[..., 0], 1e-12), 0, 1), 0.0)
amp_walker, far_flux = walker_amplification(
    walk["flux_z"], walk["flux_plane"], solid, shape_s, NU, walk["valid_faces"]
)
toc_walkers = time.perf_counter() - tic

# stiffness from measured conductance: harmonic deficit mapped to the elastic one
conductance_plain = 1.0
deficit = 1.0 - walk["conductance"] / conductance_plain
elastic_over_harmonic = 3.0 * (1.0 - NU) * (9.0 + 5.0 * NU) / (2.0 * (7.0 - 5.0 * NU)) / 1.5
k0 = E * (1.0 - deficit * elastic_over_harmonic)
print(f"conductance {walk['conductance']:.4f} -> stiffness k0 {k0:.4f}")

# ------------------------------------- min-cut ---------------------------------------
tic = time.perf_counter()
graph = build_cut_graph(solid, amp, strength, h, lateral=LATERAL)
cut = solve_min_cut(graph)
toc_cut = time.perf_counter() - tic
print(f"load bound {cut['load_bound']:.4f}, cut faces {len(cut['cut_rows'])}")

# map cut edges back to voxels and rebuild per-face amplification and strength
flat_of_id = np.flatnonzero(solid.ravel())
flat_u = flat_of_id[cut["cut_rows"]]
flat_v = flat_of_id[cut["cut_cols"]]
amp_flat = amp.ravel()
strength_flat = strength.ravel()
axis_drive = np.where(np.abs(flat_u - flat_v) == 1, 1.0, LATERAL)
amp_faces = 0.5 * (amp_flat[flat_u] + amp_flat[flat_v]) * axis_drive
strength_faces = 0.5 * (strength_flat[flat_u] + strength_flat[flat_v])

# ----------------------------------- unzip response ----------------------------------
area_void_mid = 1.0 - area.min()
unzip = unzip_response(
    amp_faces, strength_faces, h**2, area_void_mid, k0, E, NU, TOUGHNESS
)
f_break, u_break = unzip["f_break"], unzip["u_break"]
peak_idx = np.argmax(f_break)
print(f"initiation load {f_break[0]:.4f}, peak load {f_break[peak_idx]:.4f}")
print(
    f"elapsed geometry {toc_geometry:.2f} s, walkers {toc_walkers:.2f} s "
    f"({walk['steps_per_iteration']:.1f} steps/iteration), min-cut {toc_cut:.2f} s"
)

# ----------------------------------- postprocessing ----------------------------------
# equilibrium path: linear ramp, break-event sequence, terminal separation
u_curve = np.concatenate([[0.0], u_break, [u_break[-1]]])
f_curve = np.concatenate([[0.0], f_break, [0.0]])

# fracture pattern: cut-face centers with their breaking order
coords_u = np.stack(np.unravel_index(flat_u, solid.shape), axis=1)
coords_v = np.stack(np.unravel_index(flat_v, solid.shape), axis=1)
face_centers = (coords_u + coords_v + 1.0) / 2.0 * h
break_rank = np.empty(len(unzip["order"]))
break_rank[unzip["order"]] = np.arange(len(unzip["order"]))

broken = np.zeros(solid.shape)
broken.ravel()[flat_u] = 1.0
broken.ravel()[flat_v] = 1.0

if not args.book:
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(u_curve, f_curve, "k")
    ax.set_xlabel("u")
    ax.set_ylabel("F")

    # mid-plane slice through the void: amplification field and cut voxels
    mid = RESOLUTION // 2
    ax2 = fig.add_subplot(1, 3, 2)
    cmap = "rainbow_desaturated" if "rainbow_desaturated" in plt.colormaps() else "turbo"
    im = ax2.imshow(
        np.where(solid[:, mid, :], kt[:, mid, :], np.nan).T,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=cmap,
    )
    bx, bz = np.nonzero(broken[:, mid, :])
    ax2.plot((bx + 0.5) * h, (bz + 0.5) * h, "k.", markersize=2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    fig.colorbar(im, ax=ax2)

    ax3d = fig.add_subplot(1, 3, 3, projection="3d")
    sc = ax3d.scatter(
        face_centers[:, 0],
        face_centers[:, 1],
        face_centers[:, 2],
        c=break_rank,
        cmap="turbo",
        s=4,
    )
    ax3d.set_zlim(0, 1)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(u_curve, f_curve, "k")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RGB_PDF_DIR / "fracture_percolation_fu.pdf")
    save_csv(CSV_DIR / "fracture_percolation_fu.csv", u=u_curve, f=f_curve)

    out = RESULTS_DIR / "fracture_percolation.npz"
    np.savez(
        out,
        u=u_curve,
        f=f_curve,
        kt=kt,
        amp=amp,
        amp_walker=amp_walker,
        broken=broken,
        break_rank=break_rank,
        face_centers=face_centers,
        k0=k0,
        load_bound=cut["load_bound"],
    )
    save_vtk_voxels(
        RESULTS_DIR / "fracture_percolation.vtk",
        {"kt": kt, "broken": broken, "solid": solid.astype(float)},
        h,
    )
    print(f"\tsaved {out}")
