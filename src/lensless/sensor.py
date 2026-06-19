from enum import Enum

import numpy as np


class SensorOptions(Enum):
    RPI_HQ = "rpi_hq"
    RPI_GS = "rpi_gs"
    RPI_V2 = "rpi_v2"
    BASLER_287 = "basler_287"
    BASLER_548 = "basler_548"

    @staticmethod
    def values():
        return [dev.value for dev in SensorOptions]


class SensorParam:
    PIXEL_SIZE = "pixel_size"
    RESOLUTION = "resolution"
    DIAGONAL = "diagonal"
    COLOR = "color"
    BIT_DEPTH = "bit_depth"
    MAX_EXPOSURE = "max_exposure"
    MIN_EXPOSURE = "min_exposure"


sensor_dict = {
    SensorOptions.RPI_HQ.value: {
        SensorParam.PIXEL_SIZE: np.array([1.55e-6, 1.55e-6]),
        SensorParam.RESOLUTION: np.array([3040, 4056]),
        SensorParam.DIAGONAL: 7.857e-3,
        SensorParam.COLOR: True,
        SensorParam.BIT_DEPTH: [8, 12],
        SensorParam.MAX_EXPOSURE: 670.74,
        SensorParam.MIN_EXPOSURE: 0.02,
    },
    SensorOptions.RPI_GS.value: {
        SensorParam.PIXEL_SIZE: np.array([3.45e-6, 3.45e-6]),
        SensorParam.RESOLUTION: np.array([1088, 1456]),
        SensorParam.DIAGONAL: 6.3e-3,
        SensorParam.COLOR: True,
        SensorParam.BIT_DEPTH: [8, 10],
        SensorParam.MAX_EXPOSURE: 15534385e-6,
        SensorParam.MIN_EXPOSURE: 29e-6,
    },
    SensorOptions.RPI_V2.value: {
        SensorParam.PIXEL_SIZE: np.array([1.12e-6, 1.12e-6]),
        SensorParam.RESOLUTION: np.array([2464, 3280]),
        SensorParam.DIAGONAL: 4.6e-3,
        SensorParam.COLOR: True,
        SensorParam.BIT_DEPTH: [8],
        SensorParam.MAX_EXPOSURE: 11.76,
        SensorParam.MIN_EXPOSURE: 0.02,
    },
}


class VirtualSensor:
    def __init__(
        self,
        pixel_size,
        resolution,
        diagonal=None,
        color=True,
        bit_depth=None,
        downsample=None,
        **kwargs,
    ):
        self.resolution = resolution.copy()
        self.pixel_size = np.array(pixel_size).copy()
        self.diagonal = diagonal
        self.color = color
        self.bit_depth = bit_depth if bit_depth is not None else [8]
        if diagonal is not None:
            self.size = self.diagonal / np.linalg.norm(self.resolution) * self.resolution
        else:
            self.size = self.pixel_size * self.resolution
        self.pitch = self.size / self.resolution
        self.image_shape = self.resolution
        if self.color:
            self.image_shape = np.append(self.image_shape, 3)
        if downsample is not None:
            self.downsample(downsample)

    @classmethod
    def from_name(cls, name, downsample=None):
        if name not in SensorOptions.values():
            raise ValueError(f"Sensor {name} not supported.")
        return cls(**sensor_dict[name].copy(), downsample=downsample)

    def downsample(self, factor):
        self.pixel_size = self.pixel_size * factor
        self.pitch = self.pitch * factor
        self.resolution = (self.resolution / factor).astype(int)
        self.size = self.pixel_size * self.resolution
        self.image_shape = self.resolution
        if self.color:
            self.image_shape = np.append(self.image_shape, 3)
