import warnings

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.lensless.psf import simulate_psf_from_mask


ALIGNMENT = {"top_left": (80, 100), "height": 200}
DISPLAY_RES = [900, 1200]
ORIGINAL_ASPECT_RATIO = DISPLAY_RES[1] / DISPLAY_RES[0]
ALIGNMENT["width"] = int(ALIGNMENT["height"] * ORIGINAL_ASPECT_RATIO)
CROPPED_LENSED_SHAPE = (ALIGNMENT["height"], ALIGNMENT["width"], 3)


def force_rgb(image):
    if image.ndim == 2:
        warnings.warn("Converting image to RGB", stacklevel=2)
        return np.stack([image] * 3, axis=2)
    if image.ndim == 3:
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        return image[..., :3]
    raise ValueError("Image should be 2D or 3D")


def convert_image_to_float(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32)
    return image.astype(np.float32) / 65535.0


def resize(image, shape, interpolation=cv2.INTER_CUBIC):
    min_val = image.min()
    max_val = image.max()
    new_shape = [int(i) for i in shape[-3:-1]]
    if np.array_equal(np.array(image.shape)[-3:-1], new_shape):
        return image
    tmp = np.moveaxis(image, -1, 0)
    x = transforms.Resize(size=new_shape, antialias=True)(torch.from_numpy(tmp.copy())).numpy()
    x = np.moveaxis(x, 0, -1)
    return np.clip(x, min_val, max_val)


def get_cropped_lensed(lensed, lensless):
    cropped = resize(lensed, CROPPED_LENSED_SHAPE, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros(tuple(lensless.shape[:2]) + (3,), dtype=np.float32)
    top, left = ALIGNMENT["top_left"]
    canvas[top : top + ALIGNMENT["height"], left : left + ALIGNMENT["width"]] = cropped
    return canvas


def get_roi(image):
    top, left = ALIGNMENT["top_left"]
    return image[top : top + ALIGNMENT["height"], left : left + ALIGNMENT["width"]]


def hwc_to_chw_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()


def prepare_measurement(lensless):
    lensless = convert_image_to_float(force_rgb(np.array(lensless)))
    return torch.rot90(hwc_to_chw_tensor(lensless), dims=(-2, -1), k=2)


def prepare_target(lensed, measurement_shape):
    lensed = convert_image_to_float(force_rgb(np.array(lensed)))
    dummy_lensless = np.zeros((measurement_shape[-2], measurement_shape[-1], 3), dtype=np.float32)
    return hwc_to_chw_tensor(get_cropped_lensed(lensed, dummy_lensless))


def prepare_psf(mask_vals):
    psf = simulate_psf_from_mask(mask_vals)
    if psf.ndim == 4:
        psf = psf.squeeze(0)
    if psf.shape[-1] in (1, 3):
        psf = psf.permute(2, 0, 1)
    return psf.float()


def get_dataset_object(lensed, lensless, mask_vals):
    measurement = prepare_measurement(lensless)
    target = prepare_target(lensed, measurement.shape)
    psf = prepare_psf(mask_vals)
    return target, measurement, psf
