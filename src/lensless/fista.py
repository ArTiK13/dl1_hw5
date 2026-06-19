import math

import torch
from torch import nn

from src.lensless.fft import (
    adjoint_operator,
    forward_operator,
    padded_shape_from_measurement,
    psf_to_otf,
)
from src.lensless.tv import divergence, gradient


class FISTAReconstructor(nn.Module):
    def __init__(
        self,
        num_iterations=100,
        lambda_tv=2e-4,
        lipschitz_power_iterations=20,
        lipschitz_safety_factor=1.01,
        tv_prox_inner_iterations=20,
        tv_prox_tau=0.99 / math.sqrt(8),
        tv_prox_sigma=0.99 / math.sqrt(8),
        tv_prox_theta=1.0,
        eps=1e-12,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.lambda_tv = lambda_tv
        self.lipschitz_power_iterations = lipschitz_power_iterations
        self.lipschitz_safety_factor = lipschitz_safety_factor
        self.tv_prox_inner_iterations = tv_prox_inner_iterations
        self.tv_prox_tau = tv_prox_tau
        self.tv_prox_sigma = tv_prox_sigma
        self.tv_prox_theta = tv_prox_theta
        self.eps = eps

    def _lipschitz(self, measurement, otf, padded_shape):
        x = torch.randn(
            measurement.shape[0],
            measurement.shape[1],
            *padded_shape,
            device=measurement.device,
            dtype=measurement.dtype,
        )
        x = x / (x.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + self.eps)
        for _ in range(self.lipschitz_power_iterations):
            y = adjoint_operator(
                forward_operator(x, otf, measurement.shape[-2:]), otf, padded_shape
            )
            norm = y.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + self.eps
            x = y / norm
        y = adjoint_operator(
            forward_operator(x, otf, measurement.shape[-2:]), otf, padded_shape
        )
        val = (x * y).flatten(1).sum(dim=1).amax()
        return val * self.lipschitz_safety_factor + self.eps

    def _tv_nonnegative_prox(self, z, weight):
        x = z.clamp_min(0.0)
        x_bar = x
        p = torch.zeros(
            x.shape[0], x.shape[1], 2, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
        )
        for _ in range(self.tv_prox_inner_iterations):
            p = p + self.tv_prox_sigma * gradient(x_bar)
            p = p.clamp(min=-weight, max=weight)
            x_old = x
            x_tilde = x + self.tv_prox_tau * divergence(p)
            x = ((x_tilde + self.tv_prox_tau * z) / (1.0 + self.tv_prox_tau)).clamp_min(0.0)
            x_bar = x + self.tv_prox_theta * (x - x_old)
        return x

    def forward(self, measurement, psf):
        padded_shape = padded_shape_from_measurement(measurement)
        otf = psf_to_otf(psf.to(measurement), padded_shape)
        if otf.shape[0] == 1 and measurement.shape[0] > 1:
            otf = otf.expand(measurement.shape[0], -1, -1, -1)
        ct_b = adjoint_operator(measurement, otf, padded_shape)
        lipschitz = self._lipschitz(measurement, otf, padded_shape)

        x = torch.zeros_like(ct_b)
        y = x
        t = measurement.new_tensor(1.0)
        for _ in range(self.num_iterations):
            grad = adjoint_operator(
                forward_operator(y, otf, measurement.shape[-2:]), otf, padded_shape
            ) - ct_b
            x_next = self._tv_nonnegative_prox(y - grad / lipschitz, self.lambda_tv / lipschitz)
            t_next = (1.0 + torch.sqrt(1.0 + 4.0 * t.square())) / 2.0
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
            x = x_next
            t = t_next

        top = (padded_shape[0] - measurement.shape[-2]) // 2
        left = (padded_shape[1] - measurement.shape[-1]) // 2
        return x[..., top : top + measurement.shape[-2], left : left + measurement.shape[-1]]
