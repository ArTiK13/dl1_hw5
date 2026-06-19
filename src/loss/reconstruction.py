from torch import nn
import torch.nn.functional as F

from src.utils.io_utils import ROOT_PATH, extract_roi
import lpips


class ReconstructionLoss(nn.Module):
    def __init__(self, lpips_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.mse_weight = mse_weight
        if lpips_weight != 0:
            self.lpips = lpips.LPIPS(net="vgg")
        else:
            self.lpips = None

    def forward(self, prediction, target, **batch):
        pred_roi = extract_roi(prediction)
        target_roi = extract_roi(target)
        mse_loss = F.mse_loss(pred_roi, target_roi)
        if self.lpips_weight == 0:
            lpips_loss = mse_loss.new_tensor(0.0)
        else:
            lpips_pred = pred_roi.clamp(0.0, 1.0) * 2.0 - 1.0
            lpips_target = target_roi.clamp(0.0, 1.0) * 2.0 - 1.0
            lpips_loss = self.lpips(lpips_pred, lpips_target).mean()
        loss = self.mse_weight * mse_loss + self.lpips_weight * lpips_loss
        return {"loss": loss, "mse_loss": mse_loss, "lpips_loss": lpips_loss}
