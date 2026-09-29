import re
from pathlib import Path

import numpy as np
import torch

from helper import SETTINGS, export, shape_name

BASE_DIR = Path(__file__).parent
SCAN_DIR = (BASE_DIR / "../../../external_data/ctscans").resolve()  # uint8 scans from convert_ctscans.py
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/ctscans").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(0)
rect_rng = rng.spawn(1)[0]  # rectangles draw separately, so the squares stay the same

# -------------------------------------- settings -------------------------------------
MARGIN = 8  # voxels kept clear of the core surface
END_BRIGHTNESS = 0.8  # slices dimmer than this fraction of the core median are end caps
STEP = 6  # jump in mean core gray value between 5-slice windows that marks a stitch

# --------------------------------------- helper --------------------------------------
def detect_core(data):
    air, rock = np.percentile(data[::8, ::8, ::8], [1, 99])
    thresh = air + 0.25 * (rock - air)  # air is clipped to 0, so the midpoint would fall inside the rock
    zmid = data.shape[2] // 2
    mask = (data[:, :, zmid - 50 : zmid + 50] > thresh).mean(axis=2) > 0.5
    x, y = np.nonzero(mask)
    center = (x.mean(), y.mean())
    radius = np.sqrt(mask.sum() / np.pi)

    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij")
    disc = (x - center[0]) ** 2 + (y - center[1]) ** 2 < (0.9 * radius) ** 2
    core = data[disc]
    brightness = np.median(core[::5], axis=0) - air
    ids = np.nonzero(brightness >= END_BRIGHTNESS * np.median(brightness))[0]
    zmin = ids[0] + MARGIN  # walk in from both ends, so darker rock layers inside the core are kept
    zmax = ids[-1] - MARGIN
    zrange = (int(zmin), int(zmax))

    # multi-part scans jump in brightness where the parts meet; vertical crops stay within one part
    mean = core[::5].mean(axis=0)
    z = np.arange(zrange[0] + 5, zrange[1] - 5)
    jump = np.array([mean[i : i + 5].mean() - mean[i - 5 : i].mean() for i in z])
    stitches = [int(i) for i in z[np.abs(jump) > STEP]]
    if data.shape[0] == 666:  # vg-data scanner: equal sub-scans of ~420 binned slices, jumps as small as rock layering
        parts = round(data.shape[2] / 420)
        stitches += [round(k * data.shape[2] / parts) for k in range(1, parts)]
    stitches = sorted(s for s in stitches if zrange[0] < s < zrange[1])
    bounds = [zrange[0]] + [s for i, s in enumerate(stitches) if i == 0 or s - stitches[i - 1] > 10] + [zrange[1]]
    segments = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    return center, radius - MARGIN, zrange, segments


def fits_horizontal(radius, size):
    return (size[0] / 2) ** 2 + (size[1] / 2) ** 2 <= radius**2


def fits_vertical(radius, segments, size):
    return size[0] / 2 <= radius and any(b - a >= size[1] for a, b in segments)


def sample_horizontal(center, radius, zrange, size, generator):
    half = np.array(size) / 2
    while True:
        dx, dy = generator.uniform(-radius, radius, 2)
        if (abs(dx) + half[0]) ** 2 + (abs(dy) + half[1]) ** 2 <= radius**2:
            break
    x0 = int(center[0] + dx - half[0])
    y0 = int(center[1] + dy - half[1])
    z = int(generator.integers(zrange[0], zrange[1] + 1))
    return 2, z, x0, y0


def sample_vertical(center, radius, segments, size, generator):
    """size is (lateral, along the core axis)."""
    half = size[0] / 2
    axis = int(generator.integers(2))
    offset = generator.uniform(-1, 1) * np.sqrt(radius**2 - half**2)
    chord = np.sqrt(radius**2 - offset**2)
    lateral = generator.uniform(-1, 1) * (chord - half)
    slice_idx = int(center[axis] + offset)
    x0 = int(center[1 - axis] + lateral - half)
    fits = [(a, b) for a, b in segments if b - a >= size[1]]
    lengths = np.array([b - a - size[1] + 1 for a, b in fits])
    a, b = fits[generator.choice(len(fits), p=lengths / lengths.sum())]
    z0 = int(generator.integers(a, b - size[1] + 2))
    return axis, slice_idx, x0, z0


def crop(data, slice_axis, slice_idx, x0, y0, size):
    if slice_axis == 2:
        image = data[x0 : x0 + size[0], y0 : y0 + size[1], slice_idx]
    elif slice_axis == 0:
        image = data[slice_idx, x0 : x0 + size[0], y0 : y0 + size[1]]
    else:
        image = data[x0 : x0 + size[0], slice_idx, y0 : y0 + size[1]]
    return image


# ----------------------------------- preprocessing -----------------------------------
geometries = {}
index = []
for file in sorted(SCAN_DIR.glob("*.npy")):
    scan = file.stem
    prefix = re.match(r"[A-Za-z]+(-[A-Za-z]{2,})*", scan)  # type from the letter prefix, TUM-3-2 -> TUM
    kind = (prefix.group() if prefix else scan).replace("-", "_")
    data = np.load(file)

    center, radius, zrange, segments = detect_core(data)
    print(f"{scan}: radius {radius:.0f}, z {zrange}, parts {segments}")

    for aspect in SETTINGS["aspect_ratios"]:
        generator = rng if aspect == 1 else rect_rng
        for res in SETTINGS["resolutions"]:
            images = geometries.setdefault((kind, shape_name(res, aspect)), [])
            for i in range(SETTINGS["samples"]):
                # a rectangle is cut along either in-plane direction and transposed back
                transposed = aspect != 1 and bool(generator.random() < 0.5)
                size = (res // aspect, res) if transposed else (res, res // aspect)
                horizontal = i < SETTINGS["samples"] // 2
                fits = fits_horizontal(radius, size) if horizontal else fits_vertical(radius, segments, size)
                if not fits:
                    print(f"skip {scan}: {shape_name(res, aspect)} does not fit")
                    break
                if horizontal:
                    position = sample_horizontal(center, radius, zrange, size, generator)
                else:
                    position = sample_vertical(center, radius, segments, size, generator)

                image = np.array(crop(data, *position, size))
                index.append([f"{kind}_{shape_name(res, aspect)}_{len(images)}", scan, *position, transposed])
                images.append(image.T if transposed else image)

    del data

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
