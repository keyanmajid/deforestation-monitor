import numpy as np
import torch
import rasterio


@torch.no_grad()
def predict_probs(model, image, tile=256, overlap=64, batch_size=8, tta=True):
    """Forest probability map (H, W) for a full normalized image (C, H, W).

    Overlapping windows are blended with a smooth weight that favors tile centers,
    which removes seams. tta=True also averages in a horizontally flipped pass.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    _, H, W = image.shape
    stride = tile - overlap

    def starts(n):
        s = list(range(0, max(n - tile, 0) + 1, stride))
        if s[-1] + tile < n:
            s.append(n - tile)          # make sure the far edge is covered
        return s

    coords = [(r, c) for r in starts(H) for c in starts(W)]
    w1 = np.hanning(tile + 2)[1:-1]
    weight = np.outer(w1, w1).astype("float32")

    acc = np.zeros((H, W), "float32")
    wsum = np.zeros((H, W), "float32")

    for i in range(0, len(coords), batch_size):
        batch = coords[i:i + batch_size]
        x = torch.from_numpy(
            np.stack([image[:, r:r + tile, c:c + tile] for r, c in batch])
        ).to(device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            p = torch.sigmoid(model(x))
            if tta:
                p = (p + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
        p = p.float().cpu().numpy()[:, 0]
        for (r, c), pi in zip(batch, p):
            acc[r:r + tile, c:c + tile] += pi * weight
            wsum[r:r + tile, c:c + tile] += weight

    return acc / wsum


def save_raster(arr, ref_path, out_path, dtype="float32"):
    """Write a single-band raster with the same georeferencing as ref_path."""
    with rasterio.open(ref_path) as src:
        prof = src.profile
    prof.update(count=1, dtype=dtype, nodata=None, compress="deflate")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)