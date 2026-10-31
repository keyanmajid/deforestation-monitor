import numpy as np
import pystac_client
import planetary_computer as pc
from odc.stac import load

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ["B02", "B03", "B04", "B08", "B11"]   # blue, green, red, NIR, SWIR
GOOD_SCL = [4, 5, 6, 7]  # vegetation, bare soil, water, unclassified (drops clouds/shadows)


def bbox_from_center(lat, lon, km=20):
    """Square bounding box [west, south, east, north] around a point."""
    dlat = km / 2 / 111.0
    dlon = km / 2 / (111.0 * np.cos(np.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def get_composite(bbox, year, start="06-01", end="09-30",
                  max_cloud=40, resolution=10, crs="EPSG:32720"):
    """Cloud-masked median composite for one dry season.

    Returns an xarray Dataset with one 2D array per band (float32, raw DN values).
    EPSG:32720 = UTM zone 20S, which covers most of Rondonia.
    """
    catalog = pystac_client.Client.open(CATALOG_URL, modifier=pc.sign_inplace)
    items = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-{start}/{year}-{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    print(f"{year}: {len(items)} scenes found")

    ds = load(
        items,
        bands=BANDS + ["SCL"],
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        groupby="solar_day",
        chunks={"x": 1024, "y": 1024},
    )
    clear = ds["SCL"].isin(GOOD_SCL)
    optical = ds[BANDS].astype("float32").where(clear)
    return optical.median(dim="time", skipna=True).compute()


def true_color(ds, low=2, high=98):
    """RGB array (H, W, 3) scaled 0-1 with a percentile stretch, for display only."""
    rgb = np.dstack([ds["B04"].values, ds["B03"].values, ds["B02"].values])
    lo, hi = np.nanpercentile(rgb, [low, high])
    return np.clip((rgb - lo) / (hi - lo), 0, 1)