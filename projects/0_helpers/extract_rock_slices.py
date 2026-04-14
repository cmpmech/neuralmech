import matplotlib.pyplot as plt
import nrrd
import numpy as np
from pyevtk.hl import gridToVTK

# ---------------------------- slice settings ----------------------------
domain_size = 320  # 2**6*5
samples = 200  # 100

animation = False  # True
vtk = True

rocks = {
    "B-HAI-1": (30, 1050, 512, 514),
    "KAK-1": (10, 1090, 469, 460),
    "KAK-2": (50, 1080, 479, 506),
    "BM-5": (25, 1000, 469, 421),
    "BM-48": (230, 1180, 465, 439),
    # 'BM-84' : (30, 1010, 487, 451),
    "BM-105": (10, 1060, 435, 593),
    "WD-2": (130, 1100, 490, 445),
    "WD-150": (50, 1000, 453, 440),
}


# ----------------------------- select rock ------------------------------
for name in rocks.keys():
    data, _ = nrrd.read("../../external_data/" + name + ".nrrd")

    zmin, zmax, xc, yc = rocks[name]
    xmin, xmax = xc - domain_size // 2, xc + domain_size // 2
    ymin, ymax = yc - domain_size // 2, yc + domain_size // 2

# ------------------------------- clean CT -------------------------------
    data = np.array(data).astype(np.float32)
    print(f"slice before: {data.shape}")
    data = data[xmin:xmax, ymin:ymax, zmin:zmax]
    print(f"slice after: {data.shape}")

# -------------------- extract slices for animations ---------------------
    data -= np.min(data)
    data /= np.max(data)
    if animation == True:
        for i in range(data.shape[2]):
            fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
            ax.imshow(data[:, :, i].T, cmap="binary", origin="lower", vmin=0, vmax=1)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(
                f"../../results/animations/animation_frames/{name}/frame_{i}.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

# ---------------------------- export to vtk -----------------------------
    # TODO maybe move before cleaning?
    if vtk == True:
        nx, ny, nz = data.shape
        x = np.arange(0, nx + 1, dtype=np.float64)
        y = np.arange(0, ny + 1, dtype=np.float64)
        z = np.arange(0, nz + 1, dtype=np.float64)
        gridToVTK(
            f"../../results/3D/{name}",
            x,
            y,
            z,
            cellData={"scalar": np.ascontiguousarray(data)},
        )

# ---------------------------- extract slices ----------------------------
    slice_ids = np.linspace(0, data.shape[2] - 1, samples).astype(int)
    domains = data[:, :, slice_ids].transpose(2, 0, 1)
    domains -= np.min(domains)
    domains /= np.max(domains)

# -------------------------------- export --------------------------------
    np.save(f"../../data/rocks_{name}_{domain_size}.npy", domains)

# ---------------------------- postprocessing ----------------------------
    index = np.where(np.any(domains == 1, axis=(1, 2)))[0]

    fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
    ax.imshow(domains[index, :, :].T, cmap="binary", origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"../../results/rocks_{name}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

# ------------------------------- cleanup --------------------------------
    del data, domains
