import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from scipy import ndimage

VERSION = "GFC-2025-v1.13"
TILE = "00N_070W"
S8 = np.ones((3, 3), bool)


def hansen_url(layer="lossyear", version=VERSION, tile=TILE):
    return ("https://storage.googleapis.com/earthenginepartners-hansen/"
            f"{version}/Hansen_{version}_{layer}_{tile}.tif")


def load_hansen_lossyear(ref_path, url=None):
    """Hansen lossyear (0 = none, 22 = 2022, ...) resampled onto our 10 m grid."""
    url = url or hansen_url()
    with rasterio.open(ref_path) as ref:
        out = np.zeros(ref.shape, dtype="uint8")
        dst_t, dst_crs = ref.transform, ref.crs
        l, b, r, t = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
    with rasterio.open(url) as src:
        pad = 0.01
        win = from_bounds(l - pad, b - pad, r + pad, t + pad, transform=src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win)
        reproject(data, out, src_transform=src.window_transform(win), src_crs=src.crs,
                  dst_transform=dst_t, dst_crs=dst_crs, resampling=Resampling.nearest)
    return out


def evaluate_loss(lab_kept, lossyear, p_before, years=(22, 23, 24),
                  tol_px=3, min_ha=0.5, overlap=0.3):
    """Compare our confirmed patches with Hansen loss (tolerance of tol_px * 10 m)."""
    pred = lab_kept > 0
    # only count Hansen loss where our model saw forest in the first image
    ref = np.isin(lossyear, years) & (p_before > 0.5)
    ref_d = ndimage.binary_dilation(ref, S8, iterations=tol_px)
    pred_d = ndimage.binary_dilation(pred, S8, iterations=tol_px)

    pixel_prec = (pred & ref_d).sum() / max(pred.sum(), 1)
    pixel_rec = (ref & pred_d).sum() / max(ref.sum(), 1)

    # patch-level: how many of our patches / Hansen's patches overlap the other map
    ids = np.unique(lab_kept[lab_kept > 0])
    hit = ndimage.mean(ref_d, lab_kept, ids)
    patch_prec = float((hit >= overlap).mean()) if len(ids) else 0.0

    rl, rn = ndimage.label(ref, structure=S8)
    rids = np.arange(1, rn + 1)
    rarea = ndimage.sum(ref, rl, rids) * 0.01
    big = rids[rarea >= min_ha]
    rhit = ndimage.mean(pred_d, rl, big)
    patch_rec = float((rhit >= overlap).mean()) if len(big) else 0.0

    res = dict(ours_ha=pred.sum() * 0.01, hansen_ha=ref.sum() * 0.01,
               pixel_precision=pixel_prec, pixel_recall=pixel_rec,
               patch_precision=patch_prec, patch_recall=patch_rec,
               n_ours=len(ids), n_hansen=len(big))
    for k, v in res.items():
        print(f"{k:16s} {v:,.2f}")
    return res