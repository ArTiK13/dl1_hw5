from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from src.lensless.preprocessor import prepare_psf
from src.lensless.psf import psf_config_hash
from src.utils.io_utils import ROOT_PATH


HF_DATASET_REPO = "bezzam/DigiCam-Mirflickr-MultiMask-10K"


class MaskPSFCache:
    def __init__(
        self,
        mask_cache_dir=None,
        psf_cache_dir=None,
        dataset_name=HF_DATASET_REPO,
        mask_is_psf=False,
    ):
        self.mask_cache_dir = Path(mask_cache_dir or ROOT_PATH / "data/cache/digicam/masks")
        self.psf_cache_dir = Path(psf_cache_dir or ROOT_PATH / "data/cache/digicam/psf")
        self.dataset_name = dataset_name
        self.mask_is_psf = mask_is_psf
        self.mask_cache_dir.mkdir(parents=True, exist_ok=True)
        self.psf_cache_dir.mkdir(parents=True, exist_ok=True)

    def load_mask(self, mask_label):
        mask_path = self.mask_cache_dir / f"mask_{mask_label}.npy"
        if not mask_path.exists():
            downloaded = hf_hub_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                filename=f"masks/mask_{mask_label}.npy",
                local_dir=self.mask_cache_dir.parent,
            )
            mask_path = Path(downloaded)
        return np.load(mask_path)

    def psf_from_mask_label(self, mask_label):
        mask = self.load_mask(mask_label)
        if self.mask_is_psf:
            psf = torch.from_numpy(mask).float()
            if psf.ndim == 3 and psf.shape[-1] in (1, 3):
                psf = psf.permute(2, 0, 1)
            return psf
        cache_path = self.psf_cache_dir / f"mask_{mask_label}_{psf_config_hash()}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
        psf = prepare_psf(mask)
        torch.save(psf, cache_path)
        return psf
