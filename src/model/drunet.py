import torch.nn.functional as F
from torch import nn

from src.model.blocks import DownsampleBlock, ResidualStack, UpsampleBlock


DRUNET_CHANNELS = {
    "pre8": (32, 64, 128, 256),
    "post8": (32, 64, 128, 256),
    "pre4": (32, 64, 116, 128),
    "post4": (32, 64, 116, 128),
    "pre2": (16, 32, 64, 128),
    "post2": (16, 32, 64, 128),
}


class DRUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256),
        num_blocks=4,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.head = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.enc1 = ResidualStack(c1, num_blocks)
        self.down1 = DownsampleBlock(c1, c2)
        self.enc2 = ResidualStack(c2, num_blocks)
        self.down2 = DownsampleBlock(c2, c3)
        self.enc3 = ResidualStack(c3, num_blocks)
        self.down3 = DownsampleBlock(c3, c4)
        self.bottleneck = ResidualStack(c4, num_blocks)
        self.up3 = UpsampleBlock(c4, c3)
        self.dec3 = ResidualStack(c3, num_blocks)
        self.up2 = UpsampleBlock(c3, c2)
        self.dec2 = ResidualStack(c2, num_blocks)
        self.up1 = UpsampleBlock(c2, c1)
        self.dec1 = ResidualStack(c1, num_blocks)
        self.tail = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1, bias=False)

    def _match(self, x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x1 = self.enc1(self.head(x))
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.bottleneck(self.down3(x3))
        y = self._match(self.up3(x4), x3) + x3
        y = self.dec3(y)
        y = self._match(self.up2(y), x2) + x2
        y = self.dec2(y)
        y = self._match(self.up1(y), x1) + x1
        y = self.dec1(y)
        return self.tail(y)


def make_drunet(kind):
    return DRUNet(channels=DRUNET_CHANNELS[kind])
