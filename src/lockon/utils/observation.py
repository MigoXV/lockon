from __future__ import annotations

from typing import Any

import numpy as np

from lockon.protos.gym_env import gym_env_pb2
from lockon.utils.tensor import tensor_from_array


RGB_TENSOR_DTYPE = "uint8"


def _array_from_rgb_tensor(tensor: gym_env_pb2.Tensor) -> np.ndarray:
    if tensor.dtype != RGB_TENSOR_DTYPE:
        raise ValueError(f"expected rgb tensor dtype {RGB_TENSOR_DTYPE}, got {tensor.dtype}")
    return np.frombuffer(tensor.data, dtype=np.uint8).reshape(tuple(tensor.shape))


class ObservationEncoder:
    tensor_dtype: str

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def encode(self, frame_rgb: np.ndarray) -> tuple[gym_env_pb2.Tensor, dict[str, Any]]:
        raise NotImplementedError


class ObservationDecoder:
    tensor_dtype: str

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def decode(self, observation: gym_env_pb2.Tensor, info: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class RgbObservationEncoder(ObservationEncoder):
    tensor_dtype = RGB_TENSOR_DTYPE

    def encode(self, frame_rgb: np.ndarray) -> tuple[gym_env_pb2.Tensor, dict[str, Any]]:
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        if frame.ndim not in (3, 4):
            raise ValueError(f"rgb observation expects shape [H, W, C] or [N, H, W, C], got {frame.shape}")
        height, width = frame.shape[-3:-1]
        batch_size = int(frame.shape[0]) if frame.ndim == 4 else 1
        return tensor_from_array(frame), {
            "frame_codec": "rgb",
            "width": int(width),
            "height": int(height),
            "batch_size": batch_size,
        }


class RgbObservationDecoder(ObservationDecoder):
    tensor_dtype = RGB_TENSOR_DTYPE

    def decode(self, observation: gym_env_pb2.Tensor, info: dict[str, Any]) -> np.ndarray:
        return _array_from_rgb_tensor(observation).copy()


def create_observation_encoder() -> ObservationEncoder:
    return RgbObservationEncoder()


def create_observation_decoder(tensor_dtype: str) -> ObservationDecoder:
    if tensor_dtype != RGB_TENSOR_DTYPE:
        raise ValueError(f"unsupported observation tensor dtype: only {RGB_TENSOR_DTYPE} is supported")
    return RgbObservationDecoder()
