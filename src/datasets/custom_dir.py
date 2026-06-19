from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.lensless.preprocessor import prepare_measurement, prepare_psf, prepare_target
from src.utils.io_utils import load_rgb_image


class CustomDirDataset(Dataset):
    def __init__(self, root_dir, mask_is_psf=False, limit=None, indices=None):
        self.root_dir = Path(root_dir)
        self.mask_is_psf = mask_is_psf
        lensless_dir = self.root_dir / "lensless"
        self.items = sorted(lensless_dir.glob("*.png"))
        if indices is not None:
            self.items = [self.items[i] for i in indices]
        if limit is not None:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        lensless_path = self.items[index]
        image_id = lensless_path.stem
        measurement = prepare_measurement(load_rgb_image(lensless_path))
        mask_path = self.root_dir / "masks" / f"{image_id}.npy"
        mask = np.load(mask_path)
        if self.mask_is_psf:
            psf = torch.from_numpy(mask).float()
            if psf.ndim == 3 and psf.shape[-1] in (1, 3):
                psf = psf.permute(2, 0, 1)
        else:
            psf = prepare_psf(mask)
        lensed_path = self.root_dir / "lensed" / f"{image_id}.png"
        if lensed_path.exists():
            target = prepare_target(load_rgb_image(lensed_path), measurement.shape)
        else:
            target = torch.zeros_like(measurement)
        return {
            "measurement": measurement,
            "target": target,
            "psf": psf,
            "id": image_id,
            "mask_label": image_id,
        }
