import json
import numpy as np
import rasterio

OFFSET = 1000.0   # Sentinel-2 L2A processing baseline >= 04.00 (after Jan 2022)
SCALE = 10000.0


def load_reflectance(path):
    """Read a composite -> float32 array (C, H, W) of surface reflectance (0-1), NaN kept."""
    with rasterio.open(path) as src:
        dn = src.read().astype("float32")
    refl = (dn - OFFSET) / SCALE
    return np.clip(refl, 0, 1.5)


def compute_stats(refl, out_json="data/processed/stats.json"):
    """Per-band mean/std from the (training) composite, ignoring NaNs."""
    mean = np.nanmean(refl, axis=(1, 2)).tolist()
    std = np.nanstd(refl, axis=(1, 2)).tolist()
    with open(out_json, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    return mean, std


def normalize(refl, stats_json="data/processed/stats.json"):
    """Standardize with saved stats; NaN (masked cloud) pixels become 0 (the mean)."""
    with open(stats_json) as f:
        s = json.load(f)
    mean = np.array(s["mean"], dtype="float32")[:, None, None]
    std = np.array(s["std"], dtype="float32")[:, None, None]
    out = (refl - mean) / std
    return np.nan_to_num(out, nan=0.0).astype("float32")