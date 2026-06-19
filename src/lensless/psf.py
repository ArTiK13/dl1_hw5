import hashlib
import json
import sys
import types

import numpy as np
import torch

if "turtle" not in sys.modules:
    turtle_stub = types.ModuleType("turtle")
    turtle_stub.pu = None
    sys.modules["turtle"] = turtle_stub

from waveprop.devices import slm_dict

from src.lensless.sensor import VirtualSensor
from src.lensless.slm import get_intensity_psf, get_programmable_mask


DEFAULT_PSF_CONFIG = {
    "sensor": "rpi_hq",
    "slm": "adafruit",
    "downsample": 8,
    "rotate": None,
    "flipud": True,
    "use_waveprop": True,
    "vertical_shift": None,
    "horizontal_shift": None,
    "scene2mask": 0.3,
    "mask2sensor": 0.002,
    "deadspace": True,
}


def psf_config_hash(config=None):
    payload = DEFAULT_PSF_CONFIG.copy()
    if config is not None:
        payload.update(config)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:10]


def get_psf(
    vals,
    sensor="rpi_hq",
    slm="adafruit",
    downsample=8,
    rotate=None,
    flipud=False,
    use_waveprop=False,
    vertical_shift=None,
    horizontal_shift=None,
    scene2mask=None,
    mask2sensor=None,
    deadspace=True,
):
    sensor_obj = VirtualSensor.from_name(sensor, downsample=downsample)
    mask = get_programmable_mask(
        vals=vals,
        sensor=sensor_obj,
        slm_param=slm_dict[slm],
        rotate=rotate,
        flipud=flipud,
        color_filter=None,
        deadspace=deadspace,
    )

    if downsample is not None and vertical_shift is not None:
        vertical_shift = vertical_shift // downsample
    if downsample is not None and horizontal_shift is not None:
        horizontal_shift = horizontal_shift // downsample
    if vertical_shift is not None:
        mask = torch.roll(mask, vertical_shift, dims=1)
    if horizontal_shift is not None:
        mask = torch.roll(mask, horizontal_shift, dims=2)

    psf = get_intensity_psf(
        mask=mask,
        sensor=sensor_obj,
        waveprop=use_waveprop,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
    )
    psf = psf.unsqueeze(0).permute(0, 2, 3, 1)
    psf = torch.flip(psf, dims=[-3, -2])
    return psf / psf.norm()


def simulate_psf_from_mask(
    mask_vals,
    sensor="rpi_hq",
    slm="adafruit",
    downsample=8,
    rotate=None,
    flipud=True,
    use_waveprop=True,
    vertical_shift=None,
    horizontal_shift=None,
    scene2mask=0.3,
    mask2sensor=0.002,
    deadspace=True,
    revert_flip=True,
):
    vals = torch.from_numpy(np.asarray(mask_vals, dtype=np.float32))
    return get_psf(
        vals=vals,
        sensor=sensor,
        slm=slm,
        downsample=downsample,
        rotate=rotate,
        flipud=flipud,
        use_waveprop=use_waveprop,
        vertical_shift=vertical_shift,
        horizontal_shift=horizontal_shift,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        deadspace=deadspace,
    ).detach()
