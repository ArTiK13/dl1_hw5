import numpy as np
import sys
import types

import torch
from scipy.ndimage import rotate as rotate_func
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

if "turtle" not in sys.modules:
    turtle_stub = types.ModuleType("turtle")
    turtle_stub.pu = None
    sys.modules["turtle"] = turtle_stub

from waveprop.color import ColorSystem
from waveprop.devices import SLMParam
from waveprop.rs import angular_spectrum
from waveprop.slm import get_centers
from waveprop.spherical import spherical_prop


def get_ctypes(dtype, is_torch):
    if is_torch:
        if dtype in (torch.float32, torch.complex64):
            return torch.complex64
        return torch.complex128
    if dtype in (np.float32, np.complex64):
        return np.complex64
    return np.complex128


def get_programmable_mask(
    vals, sensor, slm_param, rotate=None, flipud=False, nbits=8, color_filter=None, deadspace=True
):
    use_torch = isinstance(vals, torch.Tensor)
    dtype = vals.dtype
    n_active_slm_pixels = vals.shape
    n_color_filter = int(np.prod(slm_param["color_filter"].shape[:2]))
    if color_filter is None and SLMParam.COLOR_FILTER in slm_param.keys():
        color_filter = slm_param[SLMParam.COLOR_FILTER]
        if use_torch:
            color_filter = torch.tensor(color_filter).to(vals)
    if color_filter is not None and flipud:
        color_filter = torch.flip(color_filter, dims=(0,)) if use_torch else np.flipud(color_filter)

    if use_torch:
        mask = torch.zeros((n_color_filter,) + tuple(sensor.resolution)).to(vals)
        slm_vals_flat = vals.flatten()
    else:
        mask = np.zeros((n_color_filter,) + tuple(sensor.resolution), dtype=dtype)
        slm_vals_flat = vals.reshape(-1)

    pixel_pitch = slm_param[SLMParam.PITCH]
    d1 = sensor.pitch
    if deadspace:
        centers = get_centers(n_active_slm_pixels, pixel_pitch=pixel_pitch)
        cell_h, cell_w = (slm_param[SLMParam.CELL_SIZE] / d1).astype(int)
        for i, center in enumerate(centers):
            center_pixel = (center / d1 + sensor.resolution / 2).astype(int)
            top_left = (
                center_pixel[0] - np.floor(cell_h / 2).astype(int),
                center_pixel[1] + 1 - np.floor(cell_w / 2).astype(int),
            )
            color_filter_idx = i // n_active_slm_pixels[1] % n_color_filter
            mask_val = slm_vals_flat[i] * color_filter[color_filter_idx][0]
            if use_torch:
                mask_val = mask_val.unsqueeze(-1).unsqueeze(-1)
            else:
                mask_val = mask_val[:, np.newaxis, np.newaxis]
            mask[:, top_left[0] : top_left[0] + cell_h, top_left[1] : top_left[1] + cell_w] = mask_val
    else:
        if use_torch:
            active = torch.zeros((n_color_filter,) + n_active_slm_pixels).to(vals)
        else:
            active = np.zeros((n_color_filter,) + n_active_slm_pixels, dtype=dtype)
        for i in range(n_active_slm_pixels[0]):
            row_idx = i % color_filter.shape[0]
            for j in range(n_active_slm_pixels[1]):
                col_idx = j % color_filter.shape[1]
                active[:, n_active_slm_pixels[0] - i - 1, n_active_slm_pixels[1] - j - 1] = (
                    vals[i, j] * color_filter[row_idx, col_idx]
                )
        n_active_dim = np.around(slm_param[SLMParam.PITCH] * n_active_slm_pixels / d1).astype(int)
        if use_torch:
            mask_active = transforms.functional.resize(
                active, n_active_dim, interpolation=InterpolationMode.NEAREST
            )
        else:
            mask_active = np.zeros((n_color_filter,) + tuple(n_active_dim), dtype=dtype)
            for i in range(n_color_filter):
                mask_active[i] = np.resize(active[i], n_active_dim)
        top_left = (sensor.resolution - n_active_dim) // 2
        mask[:, top_left[0] : top_left[0] + n_active_dim[0], top_left[1] : top_left[1] + n_active_dim[1]] = mask_active

    if rotate is not None:
        mask = transforms.functional.rotate(mask, angle=rotate) if use_torch else rotate_func(
            mask, axes=(2, 1), angle=rotate, reshape=False
        )
    return mask


def adafruit_sub2full(subpattern, center):
    sub_shape = subpattern.shape
    controllable_shape = (3, sub_shape[0] // 3, sub_shape[1])
    subpattern_rgb = subpattern.reshape(controllable_shape, order="F") * 255
    pattern = np.zeros((3, 128, 160), dtype=np.uint8)
    top_left = [center[0] - controllable_shape[1] // 2, center[1] - controllable_shape[2] // 2]
    pattern[:, top_left[0] : top_left[0] + controllable_shape[1], top_left[1] : top_left[1] + controllable_shape[2]] = subpattern_rgb.astype(np.uint8)
    return pattern


def full2subpattern(pattern, shape, center, slm=None):
    shape = np.array(shape)
    center = np.array(center)
    idx_1 = center[0] - shape[0] // 2
    idx_2 = center[1] - shape[1] // 2
    subpattern = pattern[:, idx_1 : idx_1 + shape[0], idx_2 : idx_2 + shape[1]] / 255.0
    if slm == "adafruit":
        subpattern = subpattern.reshape((-1, subpattern.shape[-1]), order="F")
    return subpattern


def get_intensity_psf(mask, waveprop=False, sensor=None, scene2mask=None, mask2sensor=None, color_system=None):
    if color_system is None:
        color_system = ColorSystem.rgb()
    is_torch = isinstance(mask, torch.Tensor)
    device = mask.device if is_torch else None
    ctype = get_ctypes(mask.dtype, is_torch)
    psfs = torch.zeros(mask.shape, dtype=ctype, device=device) if is_torch else np.zeros(mask.shape, dtype=ctype)
    if waveprop:
        spherical_wavefront = spherical_prop(
            in_shape=sensor.resolution,
            d1=sensor.pitch,
            wv=color_system.wv,
            dz=scene2mask,
            return_psf=True,
            is_torch=is_torch,
            device=device,
            dtype=mask.dtype,
        )
        u_in = spherical_wavefront * mask
        for i, wv in enumerate(color_system.wv):
            psfs[i], _, _ = angular_spectrum(
                u_in=u_in[i], wv=wv, d1=sensor.pitch, dz=mask2sensor, dtype=mask.dtype, device=device
            )
    else:
        psfs = mask
    return torch.square(torch.abs(psfs)) if is_torch else np.square(np.abs(psfs))
