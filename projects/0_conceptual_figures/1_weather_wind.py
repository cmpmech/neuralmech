from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cdsapi
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
DATA_PATH = BASE_DIR / "../../external_data/era5_wind.nc"

TIMES = ["6:00", "12:00", "18:00"]  # if this is updated, data needs to be downloaded
STREAM_DENSITY = 10  # streamline density
for TIMESTEP in range(3):
# ---------------------------- download data -----------------------------
    if not DATA_PATH.exists():
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": "2025",
                "month": "04",
                "day": "30",
                "time": TIMES,
                "format": "netcdf",
                "grid": [1.0, 1.0],
            },
            DATA_PATH,
        )

# ---------------------------- load & convert ----------------------------
    ds = xr.open_dataset(DATA_PATH)
    u = ds["u10"].isel(valid_time=TIMESTEP).squeeze()
    v = ds["v10"].isel(valid_time=TIMESTEP).squeeze()
    speed = np.sqrt(u**2 + v**2)

    def fix_lons(da):
        return da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby(
            "longitude"
        )

    u = fix_lons(u)
    v = fix_lons(v)
    speed = fix_lons(speed)

    # streamplot requires ascending latitudes (ERA5 is 90→-90)
    lons = u.longitude.values
    lats = u.latitude.values[::-1]
    U = u.values[::-1]
    V = v.values[::-1]
    S = speed.values[::-1]

# ---------------------------- postprocessing ----------------------------
    proj = ccrs.Mollweide()
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj}, dpi=100)
    ax.set_global()
    ax.spines["geo"].set_linewidth(0)

    ax.pcolormesh(
        speed.longitude,
        speed.latitude,
        speed.values,
        transform=ccrs.PlateCarree(),
        cmap="cividis",
        vmin=0,
        vmax=25,
        shading="auto",
        zorder=1,
    )
    ax.streamplot(
        lons,
        lats,
        U,
        V,
        transform=ccrs.PlateCarree(),
        density=STREAM_DENSITY,
        color=S,
        cmap="Greys_r",
        linewidth=0.7,
        arrowsize=0.8,
        zorder=2,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle=":", zorder=3)
    ax.gridlines(
        linewidth=0.3,
        color="gray",
        alpha=0.7,
        zorder=4,
        xlocs=range(-180, 181, 5),
        ylocs=range(-90, 91, 5),
    )
    ax.set_rasterization_zorder(3)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"climate_wind_{TIMESTEP}.png", transparent=True)
    plt.close()
