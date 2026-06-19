import math

import torch
from torch import nn

from src.lensless.fft import (
    crop_adjoint,
    crop_mask,
    padded_shape_from_measurement,
    psf_to_otf,
)
from src.lensless.tv import finite_difference_otf, gradient, psi_adjoint, soft_threshold


class ADMMReconstructor(nn.Module):
    def __init__(
        self,
        iterations=100,
        trainable=False,
        mu1=1e-4,
        mu2=1e-4,
        mu3=1e-4,
        tau=2e-4,
        eps=1e-12,
    ):
        super().__init__()
        self.iterations = iterations
        self.trainable = trainable
        self.eps = eps
        values = {"mu1": mu1, "mu2": mu2, "mu3": mu3, "tau": tau}
        if trainable:
            for name, value in values.items():
                raw = torch.full((iterations,), math.log(value), dtype=torch.float32)
                self.register_parameter(f"raw_{name}", nn.Parameter(raw))
        else:
            for name, value in values.items():
                self.register_buffer(name, torch.full((iterations,), float(value)))

    def _param(self, name, idx):
        if self.trainable:
            return getattr(self, f"raw_{name}")[idx].exp() + self.eps
        return getattr(self, name)[idx]

    def forward(self, measurement, psf):
        padded_shape = padded_shape_from_measurement(measurement)
        otf = psf_to_otf(psf.to(measurement), padded_shape)
        if otf.shape[0] == 1 and measurement.shape[0] > 1:
            otf = otf.expand(measurement.shape[0], -1, -1, -1)

        crop_ct_b = crop_adjoint(measurement, padded_shape)
        mask = crop_mask(
            measurement.shape[-2:],
            padded_shape,
            device=measurement.device,
            dtype=measurement.dtype,
        )
        diff_h, diff_v = finite_difference_otf(
            padded_shape, device=measurement.device, dtype=measurement.dtype
        )
        diff_power = diff_h.abs().square() + diff_v.abs().square()
        h_power = otf.abs().square()

        shape = (measurement.shape[0], measurement.shape[1], *padded_shape)
        x = measurement.new_zeros(shape)
        u = measurement.new_zeros(measurement.shape[0], measurement.shape[1], 2, *padded_shape)
        v = measurement.new_zeros(shape)
        w = measurement.new_zeros(shape)
        alpha1 = measurement.new_zeros(shape)
        alpha2 = measurement.new_zeros(measurement.shape[0], measurement.shape[1], 2, *padded_shape)
        alpha3 = measurement.new_zeros(shape)

        for idx in range(self.iterations):
            mu1 = self._param("mu1", idx)
            mu2 = self._param("mu2", idx)
            mu3 = self._param("mu3", idx)
            tau = self._param("tau", idx)

            hx = torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1)) * otf, dim=(-2, -1)).real
            u = soft_threshold(gradient(x) + alpha2 / mu2, tau / mu2)
            v = (alpha1 + mu1 * hx + crop_ct_b) / (mask + mu1 + self.eps)
            w = torch.clamp(alpha3 / mu3 + x, min=0.0)

            rhs = (
                (mu3 * w - alpha3)
                + psi_adjoint(mu2 * u - alpha2)
                + torch.fft.ifft2(
                    torch.fft.fft2(mu1 * v - alpha1, dim=(-2, -1)) * otf.conj(),
                    dim=(-2, -1),
                ).real
            )
            denom = mu1 * h_power + mu2 * diff_power + mu3 + self.eps
            x = torch.fft.ifft2(torch.fft.fft2(rhs, dim=(-2, -1)) / denom, dim=(-2, -1)).real

            hx_new = torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1)) * otf, dim=(-2, -1)).real
            gx_new = gradient(x)
            alpha1 = alpha1 + mu1 * (hx_new - v)
            alpha2 = alpha2 + mu2 * (gx_new - u)
            alpha3 = alpha3 + mu3 * (x - w)

        top = (padded_shape[0] - measurement.shape[-2]) // 2
        left = (padded_shape[1] - measurement.shape[-1]) // 2
        return x[..., top : top + measurement.shape[-2], left : left + measurement.shape[-1]]
