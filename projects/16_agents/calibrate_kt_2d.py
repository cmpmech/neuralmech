from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helper import compute_sdf, geometric_kt, kirsch_ligament, surface_curvatures

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
RESOLUTION = 256
RADIUS = 0.1
LOAD_AXIS = 1
SMOOTH = 2.0

# -------------------------------------- create data ----------------------------------
h = 1.0 / RESOLUTION
xc = np.linspace(h / 2, 1 - h / 2, RESOLUTION)
X, Y = np.meshgrid(xc, xc, indexing="ij")
r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
solid = r > RADIUS

# ----------------------------------- postprocessing ----------------------------------
sdf = compute_sdf(solid, h)
kappas, normal = surface_curvatures(solid, sdf, h, SMOOTH)
kt = geometric_kt(sdf, kappas, normal, LOAD_AXIS, nu=0.3)

# curvature accuracy on the ligament (exact surface curvature is 1/RADIUS)
ligament = solid & (np.abs(Y - 0.5) < h) & (r > RADIUS + 2 * h) & (r < 3 * RADIUS)
kappa_err = np.abs(kappas[..., 0][ligament] * RADIUS - 1.0)
print(f"ligament curvature: max rel err {kappa_err.max():.4f}")

# ligament profile against the exact Kirsch solution
kt_exact = kirsch_ligament(r[ligament], RADIUS)
profile_err = np.abs(kt[ligament] - kt_exact) / kt_exact
print(f"ligament Kt profile: max rel err {profile_err.max():.4f} (tol 0.05)")

# angular variation on the first solid shell against Kirsch 3 - 4 cos^2(theta)
shell = solid & (sdf < 1.5 * h) & (r < 2 * RADIUS)
c2 = normal[..., LOAD_AXIS][shell] ** 2
kt_shell_exact = kirsch_ligament(r[shell], RADIUS) * 0 + (3.0 - 4.0 * c2)
angular_err = np.abs(kt[shell] - kt_shell_exact)
print(f"surface shell angular Kt: max abs err {angular_err.max():.4f} (surface offset ~h)")

assert kappa_err.max() < 0.15, "curvature estimate off by more than 15%"
assert profile_err.max() < 0.05, "Kt ligament profile off by more than 5%"
print("all 2d anchors pass")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(np.where(solid, kt, np.nan).T, origin="lower", cmap="rainbow_desaturated" if "rainbow_desaturated" in plt.colormaps() else "turbo")
xs = r[ligament]
axs[1].plot(xs, kt[ligament], "k.")
axs[1].plot(np.sort(xs), kirsch_ligament(np.sort(xs), RADIUS), "b")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
