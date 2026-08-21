from pathlib import Path

import numpy as np
import torch

from helper import (
    compute_sdf,
    geometric_kt,
    goodier_kt,
    goodier_ligament,
    kirsch_ligament,
    build_cut_graph,
    net_section_area,
    penny_compliance,
    run_walkers,
    solve_min_cut,
    surface_curvatures,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NU = 0.3
RADIUS = 0.2
checks = []

# ----------------------------------- geometry anchors --------------------------------
data = np.load(DATA_DIR / "void_cube_64.npz")
solid = data["indicator"] > 0
h = 1.0 / 64
sdf = compute_sdf(solid, h)

fraction_exact = 1.0 - 4.0 / 3.0 * np.pi * RADIUS**3
checks.append(("solid fraction 64^3", solid.mean(), fraction_exact, 0.002))

area = net_section_area(solid, h)
checks.append(("net section mid", area[32], 1.0 - np.pi * RADIUS**2, 0.005))
checks.append(("net section end", area[0], 1.0, 1e-12))

# ------------------------------------ kirsch anchor ----------------------------------
res2d = 256
a2d = 0.1
h2d = 1.0 / res2d
xc = np.linspace(h2d / 2, 1 - h2d / 2, res2d)
X2, Y2 = np.meshgrid(xc, xc, indexing="ij")
r2 = np.sqrt((X2 - 0.5) ** 2 + (Y2 - 0.5) ** 2)
solid2d = r2 > a2d
sdf2d = compute_sdf(solid2d, h2d)
kappas2d, normal2d = surface_curvatures(solid2d, sdf2d, h2d)
kt2d = geometric_kt(sdf2d, kappas2d, normal2d, load_axis=1, nu=NU)

ligament2d = (
    solid2d & (np.abs(Y2 - 0.5) < h2d) & (r2 > a2d + 2 * h2d) & (r2 < 3 * a2d)
)
err2d = np.abs(kt2d[ligament2d] - kirsch_ligament(r2[ligament2d], a2d))
checks.append(("kirsch ligament max err", err2d.max(), 0.0, 0.1))

# ----------------------------------- goodier anchor ----------------------------------
for res in (32, 64, 128):
    data = np.load(DATA_DIR / f"void_cube_{res}.npz")
    solid3d = data["indicator"] > 0
    h3d = 1.0 / res
    sdf3d = compute_sdf(solid3d, h3d)
    kappas3d, normal3d = surface_curvatures(solid3d, sdf3d, h3d)
    kt3d = geometric_kt(sdf3d, kappas3d, normal3d, load_axis=2, nu=NU)

    xc = np.linspace(h3d / 2, 1 - h3d / 2, res)
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    ligament = (
        solid3d
        & (np.abs(Y - 0.5) < h3d)
        & (np.abs(Z - 0.5) < h3d)
        & (r > RADIUS + 2 * h3d)
        & (r < 0.45)
    )
    err = np.abs(kt3d[ligament] - goodier_ligament(r[ligament], RADIUS, NU))
    rel = err / goodier_ligament(r[ligament], RADIUS, NU)
    checks.append((f"goodier ligament {res}^3 max rel err", rel.max(), 0.0, 0.05))

checks.append(
    (f"goodier surface kt 128^3", kt3d[solid3d].max(), goodier_kt(NU), 0.06)
)

# -------------------------------------- report ---------------------------------------
failures = 0
for name, value, target, tol in checks:
    ok = abs(value - target) <= tol
    failures += not ok
    print(f"{'pass' if ok else 'FAIL'}  {name}: {value:.5f} (target {target:.5f} tol {tol})")
assert failures == 0, f"{failures} anchor checks failed"
print("all anchors pass")

# ---------------------------------- min-cut anchors ----------------------------------


res = 32
h32 = 1.0 / res
plain = np.ones((res, res, res), dtype=bool)
ones = np.ones(plain.shape)
graph = build_cut_graph(plain, ones, ones, h32)
cut = solve_min_cut(graph)
checks_cut = [("plain cube min-cut load", cut["load_bound"], 1.0, 1e-6)]

# penny compliance vs Griffith: d(added energy)/dc = G * 2 pi c with K = 2 s sqrt(c/pi)
c0, eps, E_test = 0.13, 1e-6, 1.0
dUdc = 0.5 * (penny_compliance(c0 + eps, E_test, NU) - penny_compliance(c0 - eps, E_test, NU)) / (2 * eps)
G = (2.0 * np.sqrt(c0 / np.pi)) ** 2 * (1 - NU**2) / E_test
checks_cut.append(("penny compliance vs griffith", dUdc / (G * 2 * np.pi * c0), 1.0, 1e-6))

failures = 0
for name, value, target, tol in checks_cut:
    ok = abs(value - target) <= tol
    failures += not ok
    print(f"{'pass' if ok else 'FAIL'}  {name}: {value:.6f} (target {target:.6f})")
assert failures == 0, f"{failures} min-cut checks failed"
print("all min-cut anchors pass")

# ----------------------------------- walker anchors ----------------------------------
# exercises the jump-accelerated chain end to end (statistical, ~0.4% seed scatter)
plain = np.ones((64, 64, 64), dtype=bool)
walk = run_walkers(plain, 1.0 / 64, 1_000_000, 1408, 640, device, seed=0)
checks_walk = [("plain cube conductance", walk["conductance"], 1.0, 0.012)]

data = np.load(DATA_DIR / "void_cube_64.npz")
walk = run_walkers(data["indicator"] > 0, 1.0 / 64, 1_000_000, 1408, 640, device, seed=0)
maxwell = 1.0 - 1.5 * (4.0 / 3.0 * np.pi * RADIUS**3)
checks_walk.append(("void conductance vs maxwell dilute", walk["conductance"], maxwell, 0.015))

failures = 0
for name, value, target, tol in checks_walk:
    ok = abs(value - target) <= tol
    failures += not ok
    print(f"{'pass' if ok else 'FAIL'}  {name}: {value:.4f} (target {target:.4f} tol {tol})")
assert failures == 0, f"{failures} walker checks failed"
print("all walker anchors pass")
