import csv
import json
import struct
import subprocess
import urllib.parse

import numpy as np
import torch
from scipy import ndimage

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


def sample_block(shape, res, rng, axes=(0, 1, 2)):
    """random axis-aligned res x res crop of a block volume; returns (axis, slice, corner_0, corner_1).

    Axes whose slice plane is smaller than res are never drawn. Returns None when no axis fits.
    """
    axes = [a for a in axes if min(np.delete(shape, a)) >= res]
    if not axes:
        return None
    axis = int(rng.choice(axes))
    plane = np.delete(shape, axis)
    slice_idx = int(rng.integers(shape[axis]))
    x0 = int(rng.integers(plane[0] - res + 1))
    y0 = int(rng.integers(plane[1] - res + 1))
    return axis, slice_idx, x0, y0


def crop(volume, axis, slice_idx, x0, y0, res):
    image = np.take(volume, slice_idx, axis=axis)
    return np.asarray(image[x0 : x0 + res, y0 : y0 + res])


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


def extract(volume, kind, name, resolutions, samples, rng, geometries, index, clip=None, mask=None, axes=(0, 1, 2)):
    """append `samples` random crops per resolution of one volume to geometries and index.

    `clip` maps gray values to 0-255; None keeps a binary volume as 0/255. With a `mask`,
    only crops lying entirely inside it are kept (rejection sampling). `axes` limits the slice
    normals, e.g. (0,) for cross-sections of fibres running along axis 0.
    """
    for res in resolutions:
        images = geometries.setdefault((kind, res), [])
        for _ in range(samples):
            for _ in range(1000):
                position = sample_block(volume.shape, res, rng, axes)
                if position is None or mask is None or crop(mask, *position, res).all():
                    break
            else:
                position = None
            if position is None:
                print(f"skip {name}: resolution {res} does not fit")
                break
            image = crop(volume, *position, res)
            image = normalize(image, clip) if clip is not None else (255 * (image > 0)).astype(np.uint8)
            index.append([f"{kind}_{res}_{len(images)}", name, *position])
            images.append(image)


def export(geometries, index, out_dir):
    """one uint8 tensor (N, res, res) per (type, res) plus an index csv of where each crop came from."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (kind, res), images in geometries.items():
        if not images:
            continue
        file = out_dir / f"{kind}_{res}.pt"
        torch.save(torch.from_numpy(np.stack(images)), file)
        print(f"saved {file} {(len(images), res, res)}")

    with open(out_dir / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "volume", "slice_axis", "slice_index", "corner_0", "corner_1"])
        writer.writerows(index)


def type_name(label):
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")
