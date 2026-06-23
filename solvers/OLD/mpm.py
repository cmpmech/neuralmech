import time

import numpy as np
import warp as wp

wp.init()

EYE = wp.constant(wp.mat22(1.0, 0.0, 0.0, 1.0))


@wp.func
def bspline(fx: float):
    # quadratic b-spline weights for the 3-node stencil along one axis
    a = 1.5 - fx
    b = fx - 1.0
    c = fx - 0.5
    return wp.vec3(0.5 * a * a, 0.75 - b * b, 0.5 * c * c)


@wp.kernel
def clear_grid(grid_v: wp.array(dtype=wp.vec2), grid_m: wp.array(dtype=float)):
    i = wp.tid()
    grid_v[i] = wp.vec2(0.0, 0.0)
    grid_m[i] = 0.0


@wp.kernel
def p2g(x: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
        F: wp.array(dtype=wp.mat22), C: wp.array(dtype=wp.mat22),
        grid_v: wp.array(dtype=wp.vec2), grid_m: wp.array(dtype=float),
        mu: float, lam: float, inv_dx: float, dx: float, dt: float,
        p_vol: float, p_mass: float, n_grid: int):
    p = wp.tid()
    Xp = x[p] * inv_dx
    base_x = int(Xp[0] - 0.5)
    base_y = int(Xp[1] - 0.5)
    fx = wp.vec2(Xp[0] - float(base_x), Xp[1] - float(base_y))
    wx = bspline(fx[0])
    wy = bspline(fx[1])

    # advance the deformation gradient
    Fp = (EYE + dt * C[p]) * F[p]
    J = wp.determinant(Fp)
    F[p] = Fp

    # neo-hookean kirchhoff stress folded into the apic affine momentum matrix
    tau = mu * (Fp * wp.transpose(Fp) - EYE) + lam * wp.log(J) * EYE
    stress = -(dt * p_vol * 4.0 * inv_dx * inv_dx) * tau
    affine = stress + p_mass * C[p]

    for i in range(3):
        for j in range(3):
            dpos = (wp.vec2(float(i), float(j)) - fx) * dx
            weight = wx[i] * wy[j]
            idx = (base_x + i) * n_grid + (base_y + j)
            wp.atomic_add(grid_m, idx, weight * p_mass)
            wp.atomic_add(grid_v, idx, weight * (p_mass * v[p] + affine * dpos))


@wp.kernel
def grid_op(grid_v: wp.array(dtype=wp.vec2), grid_m: wp.array(dtype=float),
            dt: float, gravity: float, n_grid: int, bound: int):
    idx = wp.tid()
    m = grid_m[idx]
    if m > 0.0:
        gi = idx // n_grid
        gj = idx - gi * n_grid
        vx = grid_v[idx][0] / m
        vy = grid_v[idx][1] / m - dt * gravity
        if gi < bound and vx < 0.0:
            vx = 0.0
        if gi >= n_grid - bound and vx > 0.0:
            vx = 0.0
        if gj < bound and vy < 0.0:
            vy = 0.0
        if gj >= n_grid - bound and vy > 0.0:
            vy = 0.0
        grid_v[idx] = wp.vec2(vx, vy)


@wp.kernel
def g2p(x: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
        C: wp.array(dtype=wp.mat22), grid_v: wp.array(dtype=wp.vec2),
        inv_dx: float, dx: float, dt: float, n_grid: int, bound: int):
    p = wp.tid()
    Xp = x[p] * inv_dx
    base_x = int(Xp[0] - 0.5)
    base_y = int(Xp[1] - 0.5)
    fx = wp.vec2(Xp[0] - float(base_x), Xp[1] - float(base_y))
    wx = bspline(fx[0])
    wy = bspline(fx[1])

    new_v = wp.vec2(0.0, 0.0)
    new_C = wp.mat22(0.0, 0.0, 0.0, 0.0)
    for i in range(3):
        for j in range(3):
            dpos = (wp.vec2(float(i), float(j)) - fx) * dx
            weight = wx[i] * wy[j]
            g_v = grid_v[(base_x + i) * n_grid + (base_y + j)]
            new_v = new_v + weight * g_v
            new_C = new_C + (4.0 * inv_dx * inv_dx * weight) * wp.outer(g_v, dpos)

    v[p] = new_v
    C[p] = new_C
    lo = float(bound) * dx
    hi = 1.0 - float(bound) * dx
    new_x = x[p] + dt * new_v
    x[p] = wp.vec2(wp.clamp(new_x[0], lo, hi), wp.clamp(new_x[1], lo, hi))


class ElasticMPM:
    """MLS-MPM solver for neo-Hookean elastic bodies on a unit-square background grid.

    Separate bodies contact one another automatically through the shared grid, so no
    explicit contact model is needed. Backed by NVIDIA Warp; runs on CUDA when a device
    is available and falls back to CPU otherwise.

    Each body is seeded as a regular particle lattice, so a fixed triangle connectivity
    (``triangles``, ``tri_block``) is built once and can be deformed per frame to render
    the bodies as surfaces rather than point clouds.
    """

    def __init__(self, blocks: list, v0=None, E: float = 1e2, nu: float = 0.2,
                 rho: float = 1.0, n_grid: int = 64, gravity: float = 50.0,
                 ppc: int = 2, bound: int = 3):
        """
        blocks: list of (x0, x1, y0, y1) regions in the unit square, one per body.
        v0: optional (len(blocks), 2) initial velocity per body (defaults to zeros).
        E, nu, rho: Young's modulus, Poisson ratio and density, shared by all bodies.
        n_grid: background grid cells per side (dx = 1 / n_grid).
        gravity: downward acceleration.
        ppc: particles per cell along one axis (a ppc x ppc lattice per cell).
        bound: wall thickness in cells for the no-penetration boundary.
        """
        dx = 1.0 / n_grid
        self.dx = dx
        self.inv_dx = float(n_grid)
        self.n_grid = n_grid
        self.bound = bound
        self.gravity = gravity

        # lame parameters
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # particle volume and mass from the ppc x ppc sampling
        self.p_vol = (dx * dx) / (ppc * ppc)
        self.p_mass = self.p_vol * rho

        if v0 is None:
            v0 = np.zeros((len(blocks), 2))
        v0 = np.asarray(v0, dtype=np.float32)

        positions = []
        velocities = []
        block_id = []
        triangles = []
        tri_block = []
        start = 0
        for b, (x0, x1, y0, y1) in enumerate(blocks):
            nx = round((x1 - x0) / dx * ppc)
            ny = round((y1 - y0) / dx * ppc)
            gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
            gy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
            mesh = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
            positions.append(mesh)
            velocities.append(np.tile(v0[b], (mesh.shape[0], 1)))
            block_id.append(np.full(mesh.shape[0], b))

            # structured quad -> two triangles, local index l = ix * ny + iy
            ix, iy = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
            a = (ix * ny + iy).ravel()
            bb = ((ix + 1) * ny + iy).ravel()
            c = ((ix + 1) * ny + iy + 1).ravel()
            d = (ix * ny + iy + 1).ravel()
            tris = np.concatenate(
                [np.stack([a, bb, c], axis=1), np.stack([a, c, d], axis=1)], axis=0
            )
            triangles.append(tris + start)
            tri_block.append(np.full(tris.shape[0], b))
            start += mesh.shape[0]

        x_np = np.concatenate(positions, axis=0).astype(np.float32)
        v_np = np.concatenate(velocities, axis=0).astype(np.float32)
        self.block_id = np.concatenate(block_id, axis=0)
        self.triangles = np.concatenate(triangles, axis=0)
        self.tri_block = np.concatenate(tri_block, axis=0)
        self.rest_edge = dx / ppc
        self.n_p = x_np.shape[0]

        self.device = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
        self.x = wp.array(x_np, dtype=wp.vec2, device=self.device)
        self.v = wp.array(v_np, dtype=wp.vec2, device=self.device)
        self.F = wp.array(np.tile(np.eye(2, dtype=np.float32), (self.n_p, 1, 1)),
                          dtype=wp.mat22, device=self.device)
        self.C = wp.zeros(self.n_p, dtype=wp.mat22, device=self.device)
        self.grid_v = wp.zeros(n_grid * n_grid, dtype=wp.vec2, device=self.device)
        self.grid_m = wp.zeros(n_grid * n_grid, dtype=float, device=self.device)

    def von_mises(self) -> np.ndarray:
        """Per-particle von Mises stress (plane strain) for the current state.

        The kernels carry the Kirchhoff stress tau = mu (F F^T - I) + lam ln(J) I;
        the Cauchy stress is sigma = tau / J, with out-of-plane sigma_zz = nu tr(sigma).
        """
        F = self.F.numpy()
        J = np.linalg.det(F)
        sigma = (self.mu * (F @ np.transpose(F, (0, 2, 1)) - np.eye(2))
                 + self.lam * np.log(J)[:, None, None] * np.eye(2)) / J[:, None, None]
        sxx = sigma[:, 0, 0]
        syy = sigma[:, 1, 1]
        sxy = sigma[:, 0, 1]
        szz = self.nu * (sxx + syy)
        return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                              + (szz - sxx) ** 2 + 6.0 * sxy ** 2))

    def solve(self, dt: float, N: int, save_every: int = 20):
        """Advance N steps and snapshot the state every save_every steps.

        Returns (positions, von_mises): positions of shape (n_frames, n_p, 2) and
        per-particle von Mises stress of shape (n_frames, n_p); frame k holds step
        k * save_every.
        """
        n_grid = self.n_grid
        frames = []
        stresses = []
        wp.synchronize()
        tic = time.time()
        for s in range(N):
            if s % save_every == 0:
                frames.append(self.x.numpy().copy())
                stresses.append(self.von_mises())
            wp.launch(clear_grid, dim=n_grid * n_grid,
                      inputs=[self.grid_v, self.grid_m], device=self.device)
            wp.launch(p2g, dim=self.n_p,
                      inputs=[self.x, self.v, self.F, self.C, self.grid_v, self.grid_m,
                              self.mu, self.lam, self.inv_dx, self.dx, dt,
                              self.p_vol, self.p_mass, n_grid], device=self.device)
            wp.launch(grid_op, dim=n_grid * n_grid,
                      inputs=[self.grid_v, self.grid_m, dt, self.gravity, n_grid,
                              self.bound], device=self.device)
            wp.launch(g2p, dim=self.n_p,
                      inputs=[self.x, self.v, self.C, self.grid_v, self.inv_dx,
                              self.dx, dt, n_grid, self.bound], device=self.device)
        wp.synchronize()
        toc = time.time()
        print(f"device {self.device}, particles {self.n_p}")
        print(f"elapsed time {toc - tic:.2f} s")
        print(f"ms/step {(toc - tic) / N * 1e3:.2f}")
        return np.stack(frames, axis=0), np.stack(stresses, axis=0)
