import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cdsapi
import matplotlib.pyplot as plt
import xarray as xr

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
DATA_PATH = (BASE_DIR / "../../external_data/era5_t2m.nc").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
TIMES = ["6:00", "12:00", "18:00"]  # if this is updated, data needs to be downloaded
DOWNLOAD = False

# ------------------------------------- load data -------------------------------------
if not DATA_PATH.exists() or DOWNLOAD:
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "2m_temperature",
            "year": "2025",
            "month": "04",
            "day": "30",
            "time": TIMES,
            "format": "netcdf",
            "grid": [1.0, 1.0],
        },
        DATA_PATH,
    )

# ------------------------------------ prepare data -----------------------------------
ds = xr.open_dataset(DATA_PATH)
for timestep in range(3):
    T = ds["t2m"].isel(valid_time=timestep).squeeze() - 273.15  # K -> C

    # ERA5 uses 0-360 longitudes -> convert to -180-180
    T = T.assign_coords(longitude=(((T.longitude + 180) % 360) - 180)).sortby(
        "longitude"
    )

# ----------------------------------- postprocessing ----------------------------------
    proj = ccrs.Mollweide()  # world as an ellipse
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj}, dpi=100)
    ax.set_global()
    ax.spines["geo"].set_linewidth(0)

    ax.pcolormesh(
        T.longitude,
        T.latitude,
        T.values,
        transform=ccrs.PlateCarree(),
        cmap="magma",
        vmin=-50,
        vmax=45,
        shading="auto",
        zorder=1,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle=":", zorder=3)
    ax.gridlines(
        linewidth=0.3,
        color="gray",
        alpha=0.7,
        zorder=4,  # makes gridlines appear above
        xlocs=range(-180, 181, 5),
        ylocs=range(-90, 91, 5),
    )
    ax.set_rasterization_zorder(2)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if args.book:
        plt.savefig(
            RESULTS_DIR / f"climate_temperature_{timestep}.png", transparent=True
        )
        plt.close()
    else:
        plt.show()
