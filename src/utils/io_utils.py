import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


ROI_TOP = 80
ROI_LEFT = 100
ROI_HEIGHT = 200
ROI_WIDTH = 266


def read_json(fname):
    """
    Read the given json file.

    Args:
        fname (str): filename of the json file.
    Returns:
        json (list[OrderedDict] | OrderedDict): loaded json.
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    """
    Write the content to the given json file.

    Args:
        content (Any JSON-friendly): content to write.
        fname (str): filename of the json file.
    """
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def load_rgb_image(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)
    return arr.astype(np.float32) / 255.0


def image_to_tensor(image):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()


def load_rgb_tensor(path):
    return image_to_tensor(load_rgb_image(path))


def tensor_to_uint8_image(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    arr = tensor.numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_tensor_png(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)


def extract_roi(tensor, top=ROI_TOP, left=ROI_LEFT, height=ROI_HEIGHT, width=ROI_WIDTH):
    if tensor.shape[-2] < top + height or tensor.shape[-1] < left + width:
        return tensor
    return tensor[..., top : top + height, left : left + width]


def keep_roi(tensor, top=ROI_TOP, left=ROI_LEFT, height=ROI_HEIGHT, width=ROI_WIDTH):
    if tensor.shape[-2] < top + height or tensor.shape[-1] < left + width:
        return tensor
    result = torch.zeros_like(tensor)
    result[..., top : top + height, left : left + width] = tensor[
        ..., top : top + height, left : left + width
    ]
    return result


def image_id_from_path(path):
    return Path(path).stem
