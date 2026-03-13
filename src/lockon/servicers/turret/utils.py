from __future__ import annotations

from typing import Any

import numpy as np
from google.protobuf.struct_pb2 import Struct

from lockon.envs.turret import TurretEnv
from lockon.protos.gym_env import gym_env_pb2
from lockon.utils import array_from_tensor, tensor_from_array


def scalar_tensor(value: object, dtype: str) -> gym_env_pb2.Tensor:
    return tensor_from_array(np.asarray(value, dtype=np.dtype(dtype)))


def _sanitize_struct_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_struct_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_struct_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def info_to_struct(info: dict[str, object]) -> Struct:
    payload = Struct()
    payload.update(_sanitize_struct_value(info))
    return payload


def build_state_info(env: TurretEnv, info: dict[str, object]) -> dict[str, object]:
    state_info = {
        "qpos": info["qpos"].tolist(),
        "qvel": info["qvel"].tolist(),
        "targets": info["targets"].tolist(),
        "aim_error": float(info["aim_error"]),
        "camera_fovy_deg": float(info["camera_fovy_deg"]),
        "camera_fovx_deg": float(info["camera_fovx_deg"]),
        "fire": {"triggered": False},
    }

    bullseye_world = env.data.site_xpos[env.bullseye_site_id].copy()
    bullseye_pixel = env.world_to_pixel(bullseye_world)
    if bullseye_pixel is not None:
        state_info["bullseye_pixel"] = [int(bullseye_pixel[0]), int(bullseye_pixel[1])]

    return state_info
