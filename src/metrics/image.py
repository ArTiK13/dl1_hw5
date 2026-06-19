import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips

from src.metrics.base_metric import BaseMetric
from src.utils.io_utils import ROOT_PATH, extract_roi


def _roi_pair(prediction, target):
    return extract_roi(prediction).clamp(0.0, 1.0), extract_roi(target).clamp(0.0, 1.0)


class ReconstructionMSE(BaseMetric):
    def __call__(self, prediction, target, **batch):
        pred, target = _roi_pair(prediction, target)
        return F.mse_loss(pred, target).item()


class ReconstructionPSNR(BaseMetric):
    def __call__(self, prediction, target, **batch):
        pred, target = _roi_pair(prediction, target)
        mse = F.mse_loss(pred, target).clamp_min(1e-12)
        return (10.0 * torch.log10(1.0 / mse)).item()


class ReconstructionSSIM(BaseMetric):
    def __init__(self, name=None):
        super().__init__(name)
        self.metric = StructuralSimilarityIndexMeasure(
            data_range=1.0, gaussian_kernel=True, kernel_size=11, sigma=1.5
        )

    def __call__(self, prediction, target, **batch):
        pred, target = _roi_pair(prediction, target)
        return self.metric.to(pred.device)(pred, target).item()


class ReconstructionLPIPS(BaseMetric):
    def __init__(self, name=None):
        super().__init__(name)
        self.metric = lpips.LPIPS(net="vgg")

    def __call__(self, prediction, target, **batch):
        pred, target = _roi_pair(prediction, target)
        metric = self.metric.to(pred.device)
        return metric(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().item()
