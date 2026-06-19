import os

import cv2
import numpy as np
import torch

RPI_HQ_CAMERA_CCM_MATRIX = np.array(
    [
        [2.0659, -0.93119, -0.13421],
        [-0.11615, 1.5593, -0.44314],
        [0.073694, -0.4368, 1.3636],
    ]
)
RPI_HQ_CAMERA_BLACK_LEVEL = 256.3
SUPPORTED_BIT_DEPTH = np.array([8, 10, 12, 16])
FLOAT_DTYPES = [np.float32, np.float64]


def rgb2gray(rgb, weights=None, keepchanneldim=True):
    if torch.is_tensor(rgb):
        if rgb.shape[-1] == 3:
            weights_t = rgb.new_tensor([0.299, 0.587, 0.114])
            out = (rgb * weights_t).sum(dim=-1)
        else:
            out = rgb.mean(dim=-1)
        return out.unsqueeze(-1) if keepchanneldim else out
    if weights is None:
        weights = np.array([0.299, 0.587, 0.114])
    out = np.tensordot(rgb, weights, axes=((rgb.ndim - 1,), 0))
    return out[..., np.newaxis] if keepchanneldim else out


def get_max_val(img, nbits=None):
    if nbits is None:
        nbits = int(np.ceil(np.log2(max(float(img.max()), 1.0))))
    if nbits not in SUPPORTED_BIT_DEPTH:
        nbits = SUPPORTED_BIT_DEPTH[nbits < SUPPORTED_BIT_DEPTH][0]
    max_val = 2**nbits - 1
    if img.max() > max_val:
        max_val = 2 ** int(np.ceil(np.log2(img.max()))) - 1
    return max_val


def bayer2rgb_cc(
    img,
    nbits,
    down=None,
    blue_gain=None,
    red_gain=None,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    nbits_out=None,
):
    if nbits_out is None:
        nbits_out = nbits
    dtype = np.uint16 if nbits_out > 8 else np.uint8
    img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB).astype(np.float32)
    if down is not None:
        img = resize(img[None, ...], factor=1 / down, interpolation=cv2.INTER_CUBIC)[0]
    img = img - black_level
    if red_gain:
        img[:, :, 0] *= red_gain
    if blue_gain:
        img[:, :, 2] *= blue_gain
    img = img / (2**nbits - 1 - black_level)
    img = np.clip(img, 0, 1)
    img = (img.reshape(-1, 3, order="F") @ ccm.T).reshape(img.shape, order="F")
    img = np.clip(img, 0, 1)
    return (img * (2**nbits_out - 1)).astype(dtype)


def load_image(
    fp,
    flip=False,
    flip_ud=False,
    flip_lr=False,
    bayer=False,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    blue_gain=None,
    red_gain=None,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    back=None,
    nbits_out=None,
    as_4d=False,
    downsample=None,
    bg=None,
    return_float=False,
    shape=None,
    dtype=None,
    normalize=True,
    bgr_input=True,
    verbose=False,
):
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    if fp.endswith((".npy", ".npz")):
        img = np.load(fp)
    else:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if bayer:
        nbits = 12 if img.max() > 255 else 8
        if back:
            img = np.clip(img.astype(np.float32) - cv2.imread(back, cv2.IMREAD_UNCHANGED), 0, None)
        img = bayer2rgb_cc(
            img,
            nbits=nbits,
            blue_gain=blue_gain,
            red_gain=red_gain,
            black_level=black_level,
            ccm=ccm,
            nbits_out=nbits_out,
        )
    elif img.ndim == 3 and bgr_input:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if flip:
        img = np.flipud(np.fliplr(img))
    if flip_ud:
        img = np.flipud(img)
    if flip_lr:
        img = np.fliplr(img)
    if bg is not None:
        if bg.max() <= 1 and img.dtype not in FLOAT_DTYPES:
            bg = bg * get_max_val(img)
        img = np.clip(img - bg, 0, None)
    if as_4d:
        if img.ndim == 3:
            img = img[np.newaxis, :, :, :]
        elif img.ndim == 2:
            img = img[np.newaxis, :, :, np.newaxis]
    if downsample is not None:
        img = resize(img, factor=1 / downsample)
    if shape is not None:
        img = resize(img, shape=shape)
    if return_float:
        img = img.astype(dtype or np.float32)
        if normalize and img.max() > 0:
            img = img / img.max()
    elif dtype is not None:
        img = img.astype(dtype)
    if verbose:
        print_image_info(img)
    return img


def resize(img, factor=None, shape=None, interpolation=cv2.INTER_CUBIC):
    img_shape = np.array(img.shape)[-3:-1]
    if factor is None and shape is None:
        raise ValueError("Must specify either factor or shape")
    new_shape = tuple((img_shape * factor).astype(int)) if shape is None else shape[-3:-1]
    new_shape = [int(i) for i in new_shape]
    if np.array_equal(img_shape, new_shape):
        return img
    if img.ndim == 4:
        resized = [
            cv2.resize(frame, dsize=tuple(new_shape[::-1]), interpolation=interpolation)
            for frame in img
        ]
        out = np.stack(resized, axis=0)
    else:
        out = cv2.resize(img, dsize=tuple(new_shape[::-1]), interpolation=interpolation)
    if out.ndim == img.ndim - 1:
        out = out[..., np.newaxis]
    return out


def get_ctypes(dtype, is_torch):
    if is_torch:
        if dtype in (torch.float32, torch.complex64):
            return torch.complex64, np.complex64
        if dtype in (torch.float64, torch.complex128):
            return torch.complex128, np.complex128
    if dtype in (np.float32, np.complex64):
        return np.complex64, np.complex64
    if dtype in (np.float64, np.complex128):
        return np.complex128, np.complex128
    raise ValueError(f"Unexpected dtype: {dtype}")


def print_image_info(img):
    print(f"dimensions : {img.shape}")
    print(f"data type : {img.dtype}")
    print(f"max  : {img.max()}")
    print(f"min  : {img.min()}")
    print(f"mean : {img.mean()}")
