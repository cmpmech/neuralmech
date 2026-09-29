import csv
import json
import struct
import subprocess
import tomllib
import urllib.parse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "../settings.toml", "rb") as f:
    SETTINGS = tomllib.load(f)["geometry"]

DPMP_URL = "https://digitalporousmedia.org/api"


def dpmp_link(project, path):
    """one-time download link of a public DPMP file."""
    system = f"drp.project.published.DRP-{project}"
    url = f"{DPMP_URL}/datafiles/tapis/download/public/{system}/{urllib.parse.quote(path)}/"
    out = subprocess.run(["curl", "-sf", "-A", "Mozilla/5.0", url], capture_output=True, text=True)
    return json.loads(out.stdout)["data"]


def dpmp_files(project):
    """(sample, path) of every file in a public DPMP project."""
    url = f"{DPMP_URL}/projects/drp.project.published.DRP-{project}/tree/"
    out = subprocess.run(["curl", "-sf", "-A", "Mozilla/5.0", url], capture_output=True, text=True)
    files = []

    def walk(node, sample):
        if node["name"] == "drp.project.sample":
            sample = node["label"]
        for f in (node.get("metadata") or {}).get("file_objs") or []:
            files.append((sample, f["path"]))
        for child in node.get("children", []):
            walk(child, sample)

    walk(json.loads(out.stdout)["tree"][0], None)
    return files


def download(url, file):
    """download to file with curl, retrying on dropped connections."""
    file.parent.mkdir(parents=True, exist_ok=True)
    part = file.with_name(file.name + ".part")  # an interrupted download never looks complete
    subprocess.run(["curl", "-sfL", "--retry", "5", "-A", "Mozilla/5.0", "-o", str(part), url], check=True)
    part.rename(file)


def read_dicom(data):
    """pixels of an uncompressed little-endian DICOM slice given as bytes."""

    def unsigned(tag):
        i = data.find(tag)
        return struct.unpack("<H", data[i + 8 : i + 10])[0]

    rows, cols = unsigned(b"\x28\x00\x10\x00US"), unsigned(b"\x28\x00\x11\x00US")
    dtype = np.uint8 if unsigned(b"\x28\x00\x00\x01US") == 8 else np.uint16
    pixels = data[data.rfind(b"\xe0\x7f\x10\x00") + 12 :]
    return np.frombuffer(pixels, dtype=dtype).reshape(rows, cols)


def percentile_clip(volume, clip, stride=4):
    """gray values at the `clip` percentiles, estimated on a strided subsample."""
    return np.percentile(volume[::stride, ::stride, ::stride], clip)


def normalize(image, clip):
    image = (image.astype(np.float32) - clip[0]) / (clip[1] - clip[0])
    return np.round(255 * np.clip(image, 0, 1)).astype(np.uint8)


def shape_name(res, aspect):
    """file tag of a crop: 256 for a square, 256x64 for aspect ratio 4."""
    return f"{res}" if aspect == 1 else f"{res}x{res // aspect}"


def sample_block(shape, size, rng, axes=(0, 1, 2)):
    """random axis-aligned crop of `size` pixels from a block volume; returns
    (axis, slice, corner_0, corner_1, transposed).

    A rectangle is cut along either in-plane direction and transposed back, so that its
    long side stays first. Axes whose slice plane is too small are never drawn. Returns
    None when no axis fits.
    """
    transposed = size[0] != size[1] and bool(rng.random() < 0.5)
    size = size[::-1] if transposed else size
    axes = [a for a in axes if all(np.delete(shape, a) >= size)]
    if not axes:
        return None
    axis = int(rng.choice(axes))
    plane = np.delete(shape, axis)
    slice_idx = int(rng.integers(shape[axis]))
    x0 = int(rng.integers(plane[0] - size[0] + 1))
    y0 = int(rng.integers(plane[1] - size[1] + 1))
    return axis, slice_idx, x0, y0, transposed


def crop(volume, axis, slice_idx, x0, y0, transposed, size):
    size = size[::-1] if transposed else size
    image = np.take(volume, slice_idx, axis=axis)
    image = np.asarray(image[x0 : x0 + size[0], y0 : y0 + size[1]])
    return image.T if transposed else image


def material_mask(volume):
    """voxels inside the specimen: above the air/material midpoint, with enclosed pores filled per slice."""
    air, material = np.percentile(volume[::4, ::4, ::4], [1, 99])
    mask = volume > (air + material) / 2
    for i in range(mask.shape[0]):
        mask[i] = ndimage.binary_fill_holes(mask[i])
    return mask


def box_mask(volume, margin=0.05):
    """bounding box of the specimen, for open structures (lattices, open foams) whose pores reach the air.

    Along each axis the box spans the slices that are more than 2% solid, trimmed by `margin` of its size.
    """
    air, material = np.percentile(volume[::4, ::4, ::4], [1, 99])
    solid = volume[::2, ::2, ::2] > (air + material) / 2
    box = []
    for axis in range(3):
        fraction = solid.mean(axis=tuple(a for a in range(3) if a != axis))
        ids = 2 * np.nonzero(fraction > 0.02)[0]
        trim = int(margin * (ids[-1] - ids[0]))
        box.append(slice(ids[0] + trim, ids[-1] - trim + 1))
    mask = np.zeros(volume.shape, dtype=bool)
    mask[tuple(box)] = True
    return mask


def texture_mask(image, fov, margin):
    """fibre region of a cross-section: high local texture (smoothed Laplacian magnitude) inside `fov`.

    Resin-rich gaps and smooth background fall outside; scales are tuned for 1000 px wide slices.
    """
    scale = image.shape[0] / 1000
    texture = np.abs(ndimage.laplace(ndimage.gaussian_filter(image.astype(np.float32), 1)))
    texture = ndimage.gaussian_filter(texture, 12 * scale)
    lo, hi = np.percentile(texture[fov], [5, 95])
    region = ndimage.binary_opening((texture > (lo + hi) / 2) & fov, iterations=int(5 * scale) + 1)
    return ndimage.binary_erosion(ndimage.binary_fill_holes(region), iterations=int(margin * scale) + 1)


def extract(volume, kind, name, rng, geometries, index, clip=None, mask=None, axes=(0, 1, 2)):
    """append the crops of one volume, `samples` per resolution and aspect ratio, to
    geometries and index.

    `clip` maps gray values to 0-255; None keeps a binary volume as 0/255. With a `mask`,
    only crops lying entirely inside it are kept (rejection sampling). `axes` limits the slice
    normals, e.g. (0,) for cross-sections of fibres running along axis 0. Rectangles draw from
    a child of `rng`, so the squares stay the same when aspect ratios are added.
    """
    rect_rng = rng.spawn(1)[0]
    for aspect in SETTINGS["aspect_ratios"]:
        generator = rng if aspect == 1 else rect_rng
        for res in SETTINGS["resolutions"]:
            size = (res, res // aspect)
            images = geometries.setdefault((kind, shape_name(res, aspect)), [])
            for _ in range(SETTINGS["samples"]):
                for _ in range(1000):
                    position = sample_block(volume.shape, size, generator, axes)
                    if position is None or mask is None or crop(mask, *position, size).all():
                        break
                else:
                    position = None
                if position is None:
                    print(f"skip {name}: {shape_name(res, aspect)} does not fit")
                    break
                image = crop(volume, *position, size)
                image = normalize(image, clip) if clip is not None else (255 * (image > 0)).astype(np.uint8)
                index.append([f"{kind}_{shape_name(res, aspect)}_{len(images)}", name, *position])
                images.append(image)


def export(geometries, index, out_dir):
    """one uint8 tensor (N, R_1, R_2) per type and shape plus an index csv of where each crop came from."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (kind, shape), images in geometries.items():
        if not images:
            continue
        file = out_dir / f"{kind}_{shape}.pt"
        torch.save(torch.from_numpy(np.stack(images)), file)
        print(f"saved {file} {(len(images), *images[0].shape)}")

    with open(out_dir / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "volume", "slice_axis", "slice_index", "corner_0", "corner_1", "transposed"])
        writer.writerows(index)


def type_name(label):
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")
