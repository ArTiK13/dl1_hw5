import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.datasets.mask_cache import HF_DATASET_REPO, MaskPSFCache
from src.lensless.preprocessor import prepare_measurement, prepare_target


class DigiCamDataset(Dataset):
    def __init__(
        self,
        dataset_name=HF_DATASET_REPO,
        split="train",
        cache_dir=None,
        mask_cache_dir=None,
        psf_cache_dir=None,
        limit=None,
        indices=None,
        validation_fraction=None,
        validation_part=None,
        validation_seed=42,
    ):
        self.split = split
        self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        self.indices = list(range(len(self.dataset))) if indices is None else list(indices)
        if validation_fraction is not None:
            generator = torch.Generator().manual_seed(validation_seed)
            permutation = torch.randperm(len(self.indices), generator=generator).tolist()
            val_count = max(1, int(round(len(permutation) * validation_fraction)))
            val_set = set(permutation[:val_count])
            if validation_part == "val":
                self.indices = [self.indices[i] for i in permutation[:val_count]]
            elif validation_part == "train":
                self.indices = [idx for i, idx in enumerate(self.indices) if i not in val_set]
            else:
                raise ValueError("validation_part must be 'train' or 'val'.")
        if limit is not None:
            self.indices = self.indices[:limit]
        self.cache = MaskPSFCache(mask_cache_dir, psf_cache_dir, dataset_name)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        item = self.dataset[source_index]
        measurement = prepare_measurement(item["lensless"])
        target = prepare_target(item["lensed"], measurement.shape)
        mask_label = item["mask_label"]
        psf = self.cache.psf_from_mask_label(mask_label)
        image_id = item.get("id", f"{self.split}_{source_index:06d}")
        return {
            "measurement": measurement,
            "target": target,
            "psf": psf,
            "id": image_id,
            "mask_label": mask_label,
        }
