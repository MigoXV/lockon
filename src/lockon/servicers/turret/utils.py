from __future__ import annotations

import numpy as np
from google.protobuf.struct_pb2 import Struct

from lockon.envs.turret import TurretEnv
from lockon.protos.gym_env import gym_env_pb2


def tensor_from_array(array: np.ndarray) -> gym_env_pb2.Tensor:
    arr = np.asarray(array)
    return gym_env_pb2.Tensor(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def scalar_tensor(value: object, dtype: str) -> gym_env_pb2.Tensor:
    return tensor_from_array(np.asarray(value, dtype=np.dtype(dtype)))


def array_from_tensor(tensor: gym_env_pb2.Tensor) -> np.ndarray:
    try:
        dtype = np.dtype(tensor.dtype)
    except TypeError as exc:
        raise ValueError(f"unsupported tensor dtype: {tensor.dtype}") from exc

    item_count = int(np.prod(tensor.shape, dtype=np.int64)) if tensor.shape else 1
    expected_size = item_count * dtype.itemsize
    if len(tensor.data) != expected_size:
        raise ValueError(
            f"tensor byte size mismatch: expected {expected_size}, got {len(tensor.data)}"
        )

    return np.frombuffer(tensor.data, dtype=dtype).reshape(tuple(tensor.shape))


def info_to_struct(info: dict[str, object]) -> Struct:
    payload = Struct()
    payload.update(info)
    return payload


def build_state_info(env: TurretEnv, info: dict[str, object]) -> dict[str, object]:
    state_info = {
        "qpos": info["qpos"].tolist(),
        "qvel": info["qvel"].tolist(),
        "targets": info["targets"].tolist(),
        "aim_error": float(info["aim_error"]),
        "fire": {"triggered": False},
    }

    bullseye_world = env.data.site_xpos[env.bullseye_site_id].copy()
    bullseye_pixel = env.world_to_pixel(bullseye_world)
    if bullseye_pixel is not None:
        state_info["bullseye_pixel"] = [int(bullseye_pixel[0]), int(bullseye_pixel[1])]

    return state_info
