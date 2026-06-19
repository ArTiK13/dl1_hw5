import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.image import (
    ReconstructionLPIPS,
    ReconstructionMSE,
    ReconstructionPSNR,
    ReconstructionSSIM,
)
from src.lensless.preprocessor import prepare_target
from src.utils.io_utils import load_rgb_image, load_rgb_tensor


class ImagePairDataset(Dataset):
    def __init__(self, gt_dir, recon_dir):
        self.gt_dir = Path(gt_dir)
        self.recon_dir = Path(recon_dir)
        gt = {p.stem: p for p in self.gt_dir.glob("*.png")}
        rec = {p.stem: p for p in self.recon_dir.glob("*.png")}
        self.ids = sorted(gt.keys() & rec.keys())
        self.gt = gt
        self.rec = rec

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        prediction = load_rgb_tensor(self.rec[image_id])
        target_image = load_rgb_image(self.gt[image_id])
        if target_image.shape[:2] == prediction.shape[-2:]:
            target = load_rgb_tensor(self.gt[image_id])
        else:
            target = prepare_target(target_image, prediction.shape)
        return {
            "target": target,
            "prediction": prediction,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--recon-dir", required=True)
    args = parser.parse_args()

    dataset = ImagePairDataset(args.gt_dir, args.recon_dir)
    if len(dataset) == 0:
        raise ValueError("No matching PNG ids found.")
    metrics = [
        ReconstructionPSNR("PSNR"),
        ReconstructionSSIM("SSIM"),
        ReconstructionMSE("MSE"),
        ReconstructionLPIPS("LPIPS"),
    ]
    totals = {metric.name: 0.0 for metric in metrics}
    with torch.no_grad():
        for sample in dataset:
            batch = {
                "prediction": sample["prediction"].unsqueeze(0),
                "target": sample["target"].unsqueeze(0),
            }
            for metric in metrics:
                totals[metric.name] += metric(**batch)
    for name, value in totals.items():
        print(f"{name}: {value / len(dataset):.6f}")


if __name__ == "__main__":
    main()
