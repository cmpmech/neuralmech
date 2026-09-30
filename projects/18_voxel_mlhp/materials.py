"""resolution-independent heterogeneous materials at voxel centres, on the GPU."""

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np


def voxel_centres(nvoxels, lengths):
    """per axis voxel centre coordinates, shaped for broadcasting."""
    D = len(nvoxels)
    axes = []
    for d, (n, L) in enumerate(zip(nvoxels, lengths)):
        shape = [1] * D
        shape[d] = n
        axes.append(((cp.arange(n) + 0.5) * L / n).reshape(shape))
    return axes


def largest_component(solid):
    labels, _ = cupyx.scipy.ndimage.label(solid)
    sizes = cp.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def gyroid(nvoxels, lengths, cells=4, thickness=0.4, plate=0.05):
    """bool gyroid sheet between two solid end plates along x (2D: the slice z = 0.25).

    Floating islands only add near-singular rigid body modes, only the largest
    connected body is kept.
    """
    x = voxel_centres(nvoxels, lengths)
    z = x[2] if len(x) == 3 else 0.25
    k = 2.0 * np.pi * cells / lengths[0]
    level_set = (
        cp.sin(k * x[0]) * cp.cos(k * x[1])
        + cp.sin(k * x[1]) * cp.cos(k * z)
        + cp.sin(k * z) * cp.cos(k * x[0])
    )
    plates = (x[0] < plate * lengths[0]) | (x[0] > (1.0 - plate) * lengths[0])
    return largest_component((cp.abs(level_set) < thickness) | plates)


def inclusions(nvoxels, lengths, fraction=0.3, radius=0.08, seed=0):
    """bool union of random spheres (discs in 2D) of one radius, overlaps allowed."""
    D = len(nvoxels)
    rng = np.random.default_rng(seed)
    volume = np.prod(lengths)
    ball = np.pi * radius**2 if D == 2 else 4.0 / 3.0 * np.pi * radius**3
    count = int(np.ceil(-np.log(1.0 - fraction) * volume / ball))  # Poisson process
    centres = rng.random((count, D)) * lengths
    x = voxel_centres(nvoxels, lengths)
    h = [L / n for L, n in zip(lengths, nvoxels)]
    inside = cp.zeros(nvoxels, dtype=bool)
    for c in centres:
        box = tuple(
            slice(max(0, int((c[d] - radius) / h[d])), int((c[d] + radius) / h[d]) + 1)
            for d in range(D)
        )
        r2 = sum((x[d][(slice(None),) * d + (box[d],)] - c[d]) ** 2 for d in range(D))
        inside[box] |= r2 < radius**2
    return inside


def random_field(nvoxels, lengths, correlation=0.1, modes=128, seed=0):
    """standard Gaussian random field, squared exponential correlation (Fourier features)."""
    D = len(nvoxels)
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(modes, D)) / correlation
    phase = rng.random(modes) * 2.0 * np.pi
    x = voxel_centres(nvoxels, lengths)
    field = cp.zeros(nvoxels, dtype=cp.float32)
    for m in range(modes):
        field += cp.cos(sum(float(k[m, d]) * x[d] for d in range(D)) + float(phase[m]))
    return field * np.sqrt(2.0 / modes)


def smooth(field, width, nvoxels, lengths):
    """Gaussian smoothing with standard deviation `width` in physical units."""
    sigma = [width * n / L for n, L in zip(nvoxels, lengths)]
    return cupyx.scipy.ndimage.gaussian_filter(field.astype(cp.float32), sigma)


def two_phase(indicator, contrast):
    """coefficient 1 where the indicator is 1 and `contrast` where it is 0."""
    return (contrast + (1.0 - contrast) * indicator).astype(cp.float32)
