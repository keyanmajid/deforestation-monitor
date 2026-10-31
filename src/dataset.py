import numpy as np
import torch
from torch.utils.data import Dataset

TILE = 256


class ForestDataset(Dataset):
    """Random crops (train) or a fixed grid (val) from one image + label mask.

    region = (row_start, row_end): which horizontal band of the image to use.
    """

    def __init__(self, image, label, region, train=True, n_samples=400):
        self.img, self.lab = image, label
        self.r0, self.r1 = region
        self.train, self.n = train, n_samples
        W = image.shape[2]
        if not train:   # fixed, non-overlapping grid for reproducible validation
            self.grid = [(r, c)
                         for r in range(self.r0, self.r1 - TILE + 1, TILE)
                         for c in range(0, W - TILE + 1, TILE)]

    def __len__(self):
        return self.n if self.train else len(self.grid)

    def __getitem__(self, i):
        if self.train:
            r = np.random.randint(self.r0, self.r1 - TILE + 1)
            c = np.random.randint(0, self.img.shape[2] - TILE + 1)
        else:
            r, c = self.grid[i]
        x = self.img[:, r:r + TILE, c:c + TILE].copy()
        y = self.lab[r:r + TILE, c:c + TILE].astype("float32").copy()

        if self.train:                       # augmentation: flips + 90-degree rotations
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())[None]