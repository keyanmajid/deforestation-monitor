import numpy as np
import geopandas as gpd
import rasterio
from scipy import ndimage
from skimage.morphology import binary_opening, disk
from rasterio.features import shapes
from shapely.geometry import shape


def ndvi(refl):
    red, nir = refl[2], refl[3]          # band order: B02, B03, B04, B08, B11
    return (nir - red) / (nir + red + 1e-6)


def detect_change(p_before, p_after, refl_before, refl_after,
                  hi=0.6, lo=0.4, open_radius=2,
                  min_ha=0.5, min_ndvi_drop=0.10, pixel_ha=0.01):
    """Return (label_image, stats) for confirmed forest-loss patches."""
    cand = (p_before > hi) & (p_after < lo)
    cand = binary_opening(cand, disk(open_radius))

    lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    area = ndimage.sum(cand, lab, idx) * pixel_ha
    ndvi_drop = ndimage.mean(ndvi(refl_before) - ndvi(refl_after), lab, idx)
    prob_drop = ndimage.mean(p_before - p_after, lab, idx)

    keep = (area >= min_ha) & (ndvi_drop >= min_ndvi_drop)
    lookup = np.zeros(n + 1, dtype="int32")
    lookup[idx[keep]] = idx[keep]
    lab_kept = lookup[lab]

    stats = {int(i): dict(area_ha=float(a), ndvi_drop=float(d), prob_drop=float(p))
             for i, a, d, p, k in zip(idx, area, ndvi_drop, prob_drop, keep) if k}
    print(f"candidates {n} -> kept {len(stats)} | "
          f"{sum(s['area_ha'] for s in stats.values()):.0f} ha")
    return lab_kept, stats


def to_geodataframe(lab_kept, stats, ref_path, period):
    """Vectorize patches to polygons with attributes (WGS84)."""
    with rasterio.open(ref_path) as src:
        transform, crs = src.transform, src.crs
    rows = []
    for geom, val in shapes(lab_kept, mask=lab_kept > 0, transform=transform):
        s = stats[int(val)]
        rows.append({"geometry": shape(geom), "id": int(val), "period": period, **s})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gdf.to_crs(4326)