import torch
import torch.nn.functional as F


def padded_shape_from_measurement(measurement):
    return measurement.shape[-2] * 2, measurement.shape[-1] * 2


def centered_crop_slices(padded_shape, image_shape):
    pad_h, pad_w = padded_shape
    h, w = image_shape
    top = (pad_h - h) // 2
    left = (pad_w - w) // 2
    return top, left, slice(top, top + h), slice(left, left + w)


def crop_center(x, image_shape):
    _, _, hs, ws = centered_crop_slices(x.shape[-2:], image_shape)
    return x[..., hs, ws]


def crop_adjoint(measurement, padded_shape):
    out = measurement.new_zeros(*measurement.shape[:-2], *padded_shape)
    _, _, hs, ws = centered_crop_slices(padded_shape, measurement.shape[-2:])
    out[..., hs, ws] = measurement
    return out


def crop_mask(image_shape, padded_shape, device=None, dtype=None):
    mask = torch.zeros(1, 1, *padded_shape, device=device, dtype=dtype)
    _, _, hs, ws = centered_crop_slices(padded_shape, image_shape)
    mask[..., hs, ws] = 1.0
    return mask


def to_chw_psf(psf):
    if psf.ndim == 5 and psf.shape[1] == 1:
        psf = psf[:, 0]
    if psf.ndim == 4 and psf.shape[-1] in (1, 3):
        psf = psf.permute(0, 3, 1, 2)
    if psf.ndim == 3 and psf.shape[-1] in (1, 3):
        psf = psf.permute(2, 0, 1)
    if psf.ndim == 3:
        psf = psf.unsqueeze(0)
    if psf.shape[1] == 1:
        psf = psf.repeat(1, 3, 1, 1)
    return psf.float()


def psf_to_otf(psf, padded_shape, dc_gain=4.0):
    psf = to_chw_psf(psf)
    shifts = (-(psf.shape[-2] // 2), -(psf.shape[-1] // 2))
    pad_h = padded_shape[0] - psf.shape[-2]
    pad_w = padded_shape[1] - psf.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("PSF spatial shape must not exceed padded reconstruction shape.")
    psf = dc_gain * psf / psf.sum(dim=(-2, -1), keepdim=True)
    psf = F.pad(psf, (0, pad_w, 0, pad_h))
    psf = torch.roll(psf, shifts=shifts, dims=(-2, -1))
    return torch.fft.fft2(psf, dim=(-2, -1))


def fft_convolve(x, otf):
    return torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1)) * otf, dim=(-2, -1)).real


def fft_convolve_adjoint(x, otf):
    return torch.fft.ifft2(
        torch.fft.fft2(x, dim=(-2, -1)) * otf.conj(), dim=(-2, -1)
    ).real


def forward_operator(x, otf, image_shape):
    return crop_center(fft_convolve(x, otf), image_shape)


def adjoint_operator(measurement, otf, padded_shape):
    return fft_convolve_adjoint(crop_adjoint(measurement, padded_shape), otf)
