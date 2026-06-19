from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualStack(nn.Sequential):
    def __init__(self, channels, num_blocks=4):
        super().__init__(*[ResidualBlock(channels) for _ in range(num_blocks)])


class DownsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True),
        )


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True),
        )
