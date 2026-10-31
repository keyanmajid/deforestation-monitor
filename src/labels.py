import numpy as np
import rasterio
import pystac_client
import planetary_computer as pc
from odc.stac import load
from odc.geo.geobox import GeoBox

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TREE_COVER = 10  # ESA WorldCover class for trees


def get_forest_labels(composite_path, out_path, year=2021):
    """Binary forest mask (1 = tree cover) on the exact grid of a composite."""
    with rasterio.open(composite_path) as src:
        gb = GeoBox(src.shape, src.transform, src.crs)
        bbox = gb.geographic_extent.boundingbox.bbox  # lon/lat box for the search

    catalog = pystac_client.Client.open(CATALOG_URL, modifier=pc.sign_inplace)
    items = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    ).item_collection()
    print(f"WorldCover {year}: {len(items)} tiles")

    ds = load(items, bands=["map"], geobox=gb, resampling="nearest")
    lc = ds["map"].max("time").values          # merge tiles; nodata is 0
    forest = (lc == TREE_COVER).astype("uint8")

    with rasterio.open(
        out_path, "w", driver="GTiff", height=forest.shape[0], width=forest.shape[1],
        count=1, dtype="uint8", crs=gb.crs.to_wkt(), transform=gb.transform,
        compress="deflate",
    ) as dst:
        dst.write(forest, 1)
    return forest