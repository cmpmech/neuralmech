"""Load-percolation fracture: method library.

Fracture without solving an elastic boundary-value problem. Elasticity enters
only through closed-form anchors (Kirsch, Goodier, Maxwell, penny crack) and
stochastic walker probes. See METHOD.md for the paradigm memo.
"""

import numpy as np
import torch
from scipy import ndimage, sparse
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


# -------------------------------------------------------------------------------------
# geometry
# -------------------------------------------------------------------------------------
def compute_sdf(solid: np.ndarray, h: float) -> np.ndarray:
    """Signed distance field on voxel centers, positive inside the solid.

    The half-voxel shift places the zero level set on the voxel boundary
    between solid and void centers.
    """
    inside = ndimage.distance_transform_edt(solid)
    outside = ndimage.distance_transform_edt(~solid)
    return h * np.where(solid, inside - 0.5, -(outside - 0.5))


def net_section_area(solid: np.ndarray, h: float, axis: int = 2) -> np.ndarray:
    """Load-bearing cross-section area of every slice perpendicular to axis."""
    counts = solid.sum(axis=tuple(i for i in range(solid.ndim) if i != axis))
    return counts * h ** (solid.ndim - 1)


def surface_curvatures(
    solid: np.ndarray,
    sdf: np.ndarray,
    h: float,
    smooth: float = 2.0,
    smooth_kappa: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Principal curvatures of the nearest surface point and unit normals.

    The curvature of an EDT distance field reflects the discrete boundary
    voxels (Voronoi artifacts), not the surface. So curvatures are computed
    from a Gaussian-smoothed indicator (valid in a band around the surface),
    denoised tangentially by a band-weighted smoothing (staircase ripples
    survive the indicator smoothing), and sampled at the pulled-back nearest
    surface point x - d * n.
    Curvature is positive where the void bulges into the solid (hole, sphere).
    Returns (kappas, normal): kappas has shape (*grid, dim-1) sorted descending,
    normal has shape (*grid, dim) and points from the void into the solid.
    """
    dim = sdf.ndim
    chi = ndimage.gaussian_filter(solid.astype(np.float64), smooth)
    grad = np.stack(np.gradient(chi, h), axis=-1)
    norm = np.linalg.norm(grad, axis=-1, keepdims=True)
    band_normal = grad / np.maximum(norm, 1e-12)

    hessian = np.empty(sdf.shape + (dim, dim))
    for i in range(dim):
        gi = np.gradient(grad[..., i], h)
        for j in range(dim):
            hessian[..., i, j] = gi[j]

    # shape operator of the indicator level sets, valid near the 0.5 level set
    eye = np.eye(dim)
    projector = eye - band_normal[..., :, None] * band_normal[..., None, :]
    shape_op = projector @ hessian @ projector / np.maximum(norm[..., None], 1e-12)

    # band-weighted tangential smoothing of the shape operator itself: averaging
    # the matrices before the eigendecomposition avoids the noise-induced
    # eigenvalue splitting that biases kappa1/kappa2 apart (staircase ripples)
    weight = np.maximum(norm[..., 0], 1e-12)
    weight_smooth = ndimage.gaussian_filter(weight, smooth_kappa)
    for i in range(dim):
        for j in range(i, dim):
            entry = ndimage.gaussian_filter(shape_op[..., i, j] * weight, smooth_kappa)
            shape_op[..., i, j] = entry / weight_smooth
            shape_op[..., j, i] = shape_op[..., i, j]

    eigs = np.linalg.eigvalsh(shape_op)
    order = np.argsort(np.abs(eigs), axis=-1)
    kappa_band = np.take_along_axis(eigs, order[..., 1:], axis=-1)
    kappa_band = -np.sort(-kappa_band, axis=-1)

    # global normals from the (smoothed) distance field, reliable at any depth
    grad_sdf = np.stack(np.gradient(ndimage.gaussian_filter(sdf, 1.0), h), axis=-1)
    norm_sdf = np.linalg.norm(grad_sdf, axis=-1, keepdims=True)
    normal = grad_sdf / np.maximum(norm_sdf, 1e-12)

    # sample band curvatures at the nearest surface point (voxel index coords)
    d = np.maximum(sdf, 0.0)
    coords = np.indices(sdf.shape) - (d[None] * np.moveaxis(normal, -1, 0)) / h
    kappas = np.stack(
        [
            ndimage.map_coordinates(kappa_band[..., i], coords, order=1, mode="nearest")
            for i in range(dim - 1)
        ],
        axis=-1,
    )
    return kappas, normal


# -------------------------------------------------------------------------------------
# geometric amplification (P1)
# -------------------------------------------------------------------------------------
def goodier_kt(nu: float) -> float:
    """Stress concentration at the equator of a spherical void, remote tension."""
    return (27.0 - 15.0 * nu) / (2.0 * (7.0 - 5.0 * nu))


def goodier_pole(nu: float) -> float:
    """Stress concentration at the poles of a spherical void (compressive)."""
    return -(3.0 + 15.0 * nu) / (2.0 * (7.0 - 5.0 * nu))


def kirsch_ligament(r: np.ndarray, a: float) -> np.ndarray:
    """Exact sigma/sigma_inf on the ligament of a circular hole (2D, r >= a)."""
    return 1.0 + a**2 / (2.0 * r**2) + 3.0 * a**4 / (2.0 * r**4)


def goodier_ligament(r: np.ndarray, a: float, nu: float) -> np.ndarray:
    """Exact sigma/sigma_inf on the equatorial ligament of a spherical void."""
    c3 = (4.0 - 5.0 * nu) / (2.0 * (7.0 - 5.0 * nu))
    c5 = 9.0 / (2.0 * (7.0 - 5.0 * nu))
    return 1.0 + c3 * (a / r) ** 3 + c5 * (a / r) ** 5


def geometric_kt(
    sdf: np.ndarray,
    kappas: np.ndarray,
    normal: np.ndarray,
    load_axis: int,
    nu: float,
) -> np.ndarray:
    """Local stress concentration from SDF curvature, exact at the anchors.

    The model interpolates between the closed-form cylinder (Kirsch, shape
    s=0) and sphere (Goodier, s=1) solutions: surface value Kt(s, angle) and
    normalized excess decay w * q^(2+s) + (1-w) * q^(4+s) with q = 1/(1 + kappa1*d).
    In 2D a hole is the cylinder case (kappas has one column, s=0).
    """
    dim = sdf.ndim
    kappa1 = kappas[..., 0]
    s = np.zeros_like(kappa1)
    if dim == 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(kappa1 > 0, np.clip(kappas[..., 1] / kappa1, 0.0, 1.0), 0.0)

    # surface values, linear in shape s between the cylinder and sphere anchors
    kt_eq = 3.0 + s * (goodier_kt(nu) - 3.0)
    kt_pole = -1.0 + s * (goodier_pole(nu) + 1.0)
    c2 = normal[..., load_axis] ** 2
    kt_surf = kt_eq * (1.0 - c2) + kt_pole * c2

    # ligament decay weights from the exact profiles (Kirsch 1/4; Goodier c3/(Kt-1))
    w_sphere = (4.0 - 5.0 * nu) / (2.0 * (7.0 - 5.0 * nu)) / (goodier_kt(nu) - 1.0)
    w = 0.25 + s * (w_sphere - 0.25)

    d = np.maximum(sdf, 0.0)
    q = np.where(kappa1 > 0, 1.0 / (1.0 + np.maximum(kappa1, 0.0) * d), 0.0)
    excess = w * q ** (2.0 + s) + (1.0 - w) * q ** (4.0 + s)
    return 1.0 + (kt_surf - 1.0) * excess


# -------------------------------------------------------------------------------------
# walker probe (P2)
# -------------------------------------------------------------------------------------
def run_walkers(
    solid: np.ndarray,
    h: float,
    walkers: int,
    steps: int,
    burnin: int,
    device: torch.device,
    seed: int = 0,
    shell: int = 9,
    kmax: int = 64,
) -> dict:
    """Lattice random-walk probe of the harmonic load-flux field.

    Walkers live on solid voxels, take uniform 6-neighbor steps (blocked moves
    stay in place = reflecting boundary), are absorbed below the bottom layer
    and respawned uniformly on the solid top layer, which drives the population
    to the steady state of the discrete Laplace problem (potential 1 at top,
    0 at bottom, insulated elsewhere). Initialization follows the linear-in-z
    plain-cube steady profile to shorten burn-in. Tallies are integers, so GPU
    atomics stay deterministic under a fixed seed.

    Jump acceleration, exact in distribution: a walker whose safe radius
    k = min(floor(EDT to the void) - shell, z, res-1-z, kmax) is >= 2 cannot
    meet the void, the absorbing bottom, or the tallied top layer within k
    steps (L1 displacement <= k), so the endpoint of k constrained steps
    equals a free k-step walk, sampled in one iteration from exact binomials
    (axis counts, then +- splits) and folded at the four lateral box faces
    (blocked-move-stays reflection is exactly the half-integer mirror fold of
    a free walk). Jump paths provably stay >= shell voxels from the void, so
    the per-face tallies stay exact on faces within the shell (the only place
    they are used); net plane crossings depend only on the endpoints of a
    path and are tallied exactly for all walkers. Every rate is per lattice
    step (walker time), so the estimators are unchanged by the acceleration.

    Returns per-face net z-crossing counts (exact where valid_faces), exact
    per-plane net crossings, the absorption rate, the top-layer occupancy, and
    the conductance estimate.
    """
    res = solid.shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    solid_t = torch.from_numpy(solid.copy()).to(device)
    flat_solid = solid_t.reshape(-1)

    # safe jump length per voxel and the faces jump paths can never cross (a
    # face with a voxel closer than `shell` to the void is only ever crossed
    # by single-stepping walkers, so its tally stays exact). The four lateral
    # box faces do not limit jumps: blocked-move-stays reflection is exactly
    # the half-integer mirror fold of a free walk, so jump endpoints are
    # folded back with period 2*res. The z-walls do (k <= min(z, res-1-z)):
    # the absorbing bottom must stay out of reach, and the top layer may only
    # be touched at a jump's final step so that its occupancy tally stays an
    # exact time average.
    edt = (
        ndimage.distance_transform_edt(solid)
        if not solid.all()
        else np.full(solid.shape, np.inf)
    )
    z_idx = np.indices(solid.shape)[2]
    with np.errstate(invalid="ignore"):
        k_grid = np.clip(
            np.minimum(np.floor(edt) - shell, np.minimum(z_idx, res - 1 - z_idx)),
            0,
            kmax,
        )
    k_flat = torch.from_numpy(k_grid.reshape(-1).astype(np.float32)).to(device)
    valid_faces = (edt[:, :, :-1] < shell) | (edt[:, :, 1:] < shell)
    third = torch.full((walkers,), 1.0 / 3.0, device=device)
    half = torch.full((walkers,), 0.5, device=device)

    # initial positions: linear-in-z steady profile over solid voxels, divided
    # by the per-iteration time advance k_eff -- the stationary per-iteration
    # snapshot measure of the jump chain is (time-stationary density)/k_eff,
    # so initializing from it removes the slow snapshot redistribution
    # transient that would otherwise bias the tallies
    k_eff = torch.from_numpy(np.where(k_grid >= 2, k_grid, 1.0)).to(device)
    weights = (solid_t * (torch.arange(res, device=device) + 1.0) / k_eff).reshape(-1)
    flat = torch.multinomial(weights, walkers, replacement=True, generator=gen)
    z = flat % res
    y = (flat // res) % res
    x = flat // (res * res)

    # respawn pool: solid voxels of the top layer
    top_ids = torch.nonzero(solid_t[:, :, res - 1].reshape(-1), as_tuple=False)[:, 0]

    mx = torch.tensor([1, -1, 0, 0, 0, 0], device=device)
    my = torch.tensor([0, 0, 1, -1, 0, 0], device=device)
    mz = torch.tensor([0, 0, 0, 0, 1, -1], device=device)
    flux_z = torch.zeros(res * res * res, dtype=torch.int32, device=device)
    plane_diff = torch.zeros(res, dtype=torch.int64, device=device)
    bottom_occupancy = 0
    top_occupancy = 0
    time_total = 0

    for step in range(steps):
        k = k_flat[(x * res + y) * res + z]
        jump = k >= 2.0

        # jump endpoints: exact multinomial split of k steps over the axes,
        # exact binomial +- split per axis (k = 0 where not jumping)
        kj = torch.where(jump, k, torch.zeros_like(k))
        n1 = torch.binomial(kj, third, generator=gen)
        n2 = torch.binomial(kj - n1, half, generator=gen)
        n3 = kj - n1 - n2
        dx = (2.0 * torch.binomial(n1, half, generator=gen) - n1).long()
        dy = (2.0 * torch.binomial(n2, half, generator=gen) - n2).long()
        dz = (2.0 * torch.binomial(n3, half, generator=gen) - n3).long()

        d = torch.randint(0, 6, (walkers,), device=device, generator=gen)
        xn, yn, zn = x + mx[d], y + my[d], z + mz[d]

        absorb = ~jump & (zn < 0)
        inside = (
            (xn >= 0) & (xn < res) & (yn >= 0) & (yn < res) & (zn >= 0) & (zn < res)
        )
        xn = torch.where(inside, xn, x)
        yn = torch.where(inside, yn, y)
        zn = torch.where(inside, zn, z)
        open_move = ~jump & inside & flat_solid[(xn * res + yn) * res + zn]

        xn = torch.where(open_move, xn, x)
        yn = torch.where(open_move, yn, y)
        zn = torch.where(open_move, zn, z)
        xj = torch.remainder(x + dx, 2 * res)
        yj = torch.remainder(y + dy, 2 * res)
        xn = torch.where(jump, torch.minimum(xj, 2 * res - 1 - xj), xn)
        yn = torch.where(jump, torch.minimum(yj, 2 * res - 1 - yj), yn)
        zn = torch.where(jump, z + dz, zn)

        if step >= burnin:
            up = open_move & (d == 4)
            down = open_move & (d == 5)
            flux_z.index_add_(0, (x * res + y) * res + z, up.to(torch.int32))
            flux_z.index_add_(0, (xn * res + yn) * res + zn, -down.to(torch.int32))
            plane_diff += torch.bincount(z, minlength=res) - torch.bincount(
                zn, minlength=res
            )
            # Rao-Blackwellized absorption: a bottom-layer walker absorbs with
            # probability exactly 1/6, so occupancy/6 estimates the rate with
            # the shot noise of the realized absorption events removed
            bottom_occupancy += (z == 0).sum()
            top_occupancy += (z == res - 1).sum()
            time_total += torch.where(jump, k, torch.ones_like(k)).long().sum()

        # respawn absorbed walkers uniformly on the solid top layer (branchless,
        # to avoid a gpu->cpu sync every step)
        pick = top_ids[
            torch.randint(0, len(top_ids), (walkers,), device=device, generator=gen)
        ]
        xn = torch.where(absorb, pick // res, xn)
        yn = torch.where(absorb, pick % res, yn)
        zn = torch.where(absorb, torch.full_like(zn, res - 1), zn)
        x, y, z = xn, yn, zn

    # walker time per walker, post burn-in: every rate is per lattice step
    t_avg = float(time_total) / walkers
    rate = int(bottom_occupancy) / 6.0 / t_avg
    top_count = int(solid[:, :, res - 1].sum())
    density_top = int(top_occupancy) / t_avg / top_count
    conductance = rate * 6.0 * h / max(density_top, 1e-30)
    flux_plane = torch.cumsum(plane_diff, 0)[: res - 1].cpu().numpy() / t_avg
    return {
        "flux_z": flux_z.reshape(res, res, res).cpu().numpy() / t_avg,
        "flux_plane": flux_plane,
        "valid_faces": valid_faces,
        "rate": rate,
        "density_top": density_top,
        "conductance": conductance,
        "steps_per_iteration": t_avg / (steps - burnin),
    }


def walker_amplification(
    flux_z: np.ndarray,
    flux_plane: np.ndarray,
    solid: np.ndarray,
    s: np.ndarray,
    nu: float,
    valid_faces: np.ndarray,
    far_slabs: int = 6,
) -> tuple[np.ndarray, float]:
    """Map walker flux tallies to an elastic amplification field on z-faces.

    Concentration c = flux/far-field flux is harmonic (conductivity-like); the
    elastic excess is c_elastic - 1 = (c - 1) * M(s) with the anchor ratio
    M(s) = (Kt_eq(s) - 1)/(c_harmonic(s) - 1), where the harmonic surface
    concentration is 2 for a cylinder (s=0) and 3/2 for a sphere (s=1).
    The far-field face flux comes from the exact per-plane crossings; per-face
    concentrations are reported only where the per-face tallies are exact
    (valid_faces, the shell around the void -- zero elsewhere).
    Returns (amplification on faces, far-field flux per face).
    """
    res = solid.shape[2]
    open_face = solid[:, :, :-1] & solid[:, :, 1:]
    flux = flux_z[:, :, :-1]
    open_count = open_face.sum(axis=(0, 1))
    far_ids = list(range(2, 2 + far_slabs)) + list(range(res - 2 - far_slabs, res - 2))
    far = (flux_plane[far_ids] / open_count[far_ids]).mean()

    kt_eq = 3.0 + s * (goodier_kt(nu) - 3.0)
    c_harmonic = 2.0 - 0.5 * s
    ratio = (kt_eq - 1.0) / (c_harmonic - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(open_face, flux / far, 0.0)
    amp = 1.0 + (c - 1.0) * ratio[:, :, :-1]
    return np.where(open_face & valid_faces, amp, 0.0), float(far)


# -------------------------------------------------------------------------------------
# min-cut fracture
# -------------------------------------------------------------------------------------
def build_cut_graph(
    solid: np.ndarray,
    amp: np.ndarray,
    strength: np.ndarray,
    h: float,
    lateral: float = 0.3,
    quantum: float = 1.0e4,
) -> dict:
    """Directed capacity graph whose min-cut is the fracture surface.

    Nodes are solid voxels plus a supersource (top layer) and supersink
    (bottom layer). A face transmits load up to strength * area / amp, i.e.
    the remote load at which the local stress reaches the strength; lateral
    (x/y) faces carry the reduced driving amp * lateral. Capacities are
    quantized so the median face maps to `quantum` (scipy's flow arrays are
    int32, so the smallest z-plane capacity sum must stay below 2^31).
    """
    res = solid.shape[0]
    ids = -np.ones(solid.shape, dtype=np.int64)
    n_solid = int(solid.sum())
    ids[solid] = np.arange(n_solid)
    source, sink = n_solid, n_solid + 1

    rows, cols, caps_float, face_records = [], [], [], []
    for axis, drive in ((0, lateral), (1, lateral), (2, 1.0)):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        open_face = solid[lo] & solid[hi]
        amp_face = 0.5 * (amp[lo] + amp[hi])[open_face] * drive
        strength_face = 0.5 * (strength[lo] + strength[hi])[open_face]
        cap = strength_face * h**2 / np.maximum(amp_face, 1e-3)
        u, v = ids[lo][open_face], ids[hi][open_face]
        rows += [u, v]
        cols += [v, u]
        caps_float += [cap, cap]
        idx = np.argwhere(open_face)
        face_records.append((axis, idx, cap))

    # quantize: median z-face capacity -> quantum, clip to keep int32 flow safe
    z_caps = face_records[2][2]
    scale = quantum / np.median(z_caps)
    caps_int = [np.clip(np.round(c * scale), 1, 1e6).astype(np.int32) for c in caps_float]

    plane_sums = [
        np.clip(np.round(face_records[2][2] * scale), 1, 1e6)[
            face_records[2][1][:, 2] == k
        ].sum()
        for k in range(res - 1)
    ]
    assert min(plane_sums) < 2**31 - 1, "quantized capacities overflow int32 flow"

    big = np.int32(2**31 - 1)
    top = ids[:, :, res - 1][solid[:, :, res - 1]]
    bottom = ids[:, :, 0][solid[:, :, 0]]
    rows += [np.full(len(top), source), bottom]
    cols += [top, np.full(len(bottom), sink)]
    caps_int += [np.full(len(top), big, np.int32), np.full(len(bottom), big, np.int32)]


    graph = sparse.csr_array(
        (
            np.concatenate(caps_int),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n_solid + 2, n_solid + 2),
        dtype=np.int32,
    )
    return {
        "graph": graph,
        "source": source,
        "sink": sink,
        "scale": scale,
        "ids": ids,
        "n_solid": n_solid,
    }


def solve_min_cut(cut_graph: dict) -> dict:
    """Max-flow / min-cut: rupture-bound load, cut faces, and side labels.

    The cut is extracted as the faces from source-reachable to unreachable
    nodes in the residual graph. Returns the load bound (flow value in
    physical units), a per-voxel side label, and the cut faces as
    (i, j, k, axis, orientation) records.
    """

    graph = cut_graph["graph"]
    result = maximum_flow(graph, cut_graph["source"], cut_graph["sink"])
    residual = graph - result.flow
    residual.data = np.maximum(residual.data, 0)
    reach_ids = breadth_first_order(
        residual, cut_graph["source"], directed=True, return_predecessors=False
    )
    reachable = np.zeros(graph.shape[0], dtype=bool)
    reachable[reach_ids] = True

    coo = graph.tocoo()
    on_cut = (
        reachable[coo.row]
        & ~reachable[coo.col]
        & (coo.row != cut_graph["source"])
        & (coo.col != cut_graph["sink"])
    )
    return {
        "load_bound": result.flow_value / cut_graph["scale"],
        "reachable": reachable,
        "cut_rows": coo.row[on_cut],
        "cut_cols": coo.col[on_cut],
    }


# -------------------------------------------------------------------------------------
# load-displacement (unzipping)
# -------------------------------------------------------------------------------------
def penny_compliance(c: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Added compliance of a penny crack of radius c (unit gross area, Tada)."""
    return 16.0 * (1.0 - nu**2) * c**3 / (3.0 * E)


def unzip_response(
    amp_faces: np.ndarray,
    strength_faces: np.ndarray,
    area_faces: float,
    area_start: float,
    k0: float,
    E: float,
    nu: float,
    K_Ic: float,
) -> dict:
    """Quasi-static displacement-controlled response of the unzipping cut.

    Faces break in order of static overload amp/strength. At state k the crack
    is an effective penny of radius c_k = sqrt((area_start + k * A_face)/pi);
    the next face breaks at the smaller of the strength criterion (static
    amplification, short-crack regime) and the toughness criterion via the
    penny-crack K_I = 2 sigma sqrt(c/pi) (long-crack regime, resolution
    independent). Stiffness at state k follows the analytic penny compliance.
    Returns the event path (u, F at each break) and the stiffness history.
    """
    order = np.argsort(-amp_faces / strength_faces)
    amp_sorted = amp_faces[order]
    strength_sorted = strength_faces[order]
    n = len(order)

    area_broken = area_start + area_faces * np.arange(n)
    c = np.sqrt(area_broken / np.pi)
    stiffness = 1.0 / (1.0 / k0 + penny_compliance(c, E, nu) - penny_compliance(np.sqrt(area_start / np.pi), E, nu))

    f_strength = strength_sorted / amp_sorted
    f_tough = K_Ic / (2.0 * np.sqrt(np.maximum(c, 1e-12) / np.pi))
    f_break = np.minimum(f_strength, f_tough)
    u_break = f_break / stiffness
    return {
        "order": order,
        "f_break": f_break,
        "u_break": u_break,
        "stiffness": stiffness,
        "crack_radius": c,
        "f_strength": f_strength,
        "f_tough": f_tough,
    }


# -------------------------------------------------------------------------------------
# random material field & export
# -------------------------------------------------------------------------------------
def lognormal_field(
    shape: tuple, corr_voxels: float, sigma_log: float, rng: np.random.Generator
) -> np.ndarray:
    """Unit-median lognormal random field with Gaussian correlation length."""
    noise = rng.standard_normal(shape)
    smooth = ndimage.gaussian_filter(noise, corr_voxels)
    smooth /= smooth.std()
    return np.exp(sigma_log * smooth)


def save_vtk_voxels(path, fields: dict, h: float) -> None:
    """Write voxel fields as a legacy-VTK structured-points file (cell data)."""
    shape = next(iter(fields.values())).shape
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nload-percolation fracture\nASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {shape[0] + 1} {shape[1] + 1} {shape[2] + 1}\n")
        f.write(f"ORIGIN 0 0 0\nSPACING {h} {h} {h}\n")
        f.write(f"CELL_DATA {np.prod(shape)}\n")
        for name, field in fields.items():
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, field.transpose(2, 1, 0).reshape(-1, 1), fmt="%.5g")
