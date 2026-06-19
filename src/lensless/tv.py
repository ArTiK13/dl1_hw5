import torch


def gradient(x):
    dh = torch.roll(x, shifts=-1, dims=-1) - x
    dv = torch.roll(x, shifts=-1, dims=-2) - x
    return torch.stack((dh, dv), dim=2)


def divergence(g):
    dh = g[:, :, 0]
    dv = g[:, :, 1]
    return (dh - torch.roll(dh, shifts=1, dims=-1)) + (
        dv - torch.roll(dv, shifts=1, dims=-2)
    )


def psi_adjoint(g):
    return -divergence(g)


def soft_threshold(x, threshold):
    return torch.sign(x) * torch.clamp(x.abs() - threshold, min=0.0)


def tv_norm(x):
    return gradient(x).abs().sum(dim=(1, 2, 3, 4))


def finite_difference_otf(padded_shape, device=None, dtype=torch.float32):
    h, w = padded_shape
    kernel_h = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
    kernel_v = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
    kernel_h[..., 0, 0] = -1.0
    kernel_h[..., 0, 1 % w] = 1.0
    kernel_v[..., 0, 0] = -1.0
    kernel_v[..., 1 % h, 0] = 1.0
    return (
        torch.fft.fft2(kernel_h, dim=(-2, -1)),
        torch.fft.fft2(kernel_v, dim=(-2, -1)),
    )
