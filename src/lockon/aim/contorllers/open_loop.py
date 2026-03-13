from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lockon.aim.back_projection import backproject_to_spherical
from lockon.aim.contorllers.base import AimController, AimMetrics
from lockon.envs.turret import TurretEnv

YAW_STEP_RAD = float(TurretEnv.TARGET_STEP_SCALE[3])
PITCH_STEP_RAD = float(TurretEnv.TARGET_STEP_SCALE[4])


@dataclass(frozen=True, slots=True)
class OpenLoopMetrics(AimMetrics):
    plane_x: float
    plane_y: float
    azimuth_deg: float
    elevation_deg: float

    def as_dict(self) -> dict[str, float]:
        return {
            "plane_x": self.plane_x,
            "plane_y": self.plane_y,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
        }


def normalize_plane_coordinate(
    bullseye_pixel: list[object],
    width: int,
    height: int,
) -> tuple[float, float]:
    px = float(bullseye_pixel[0])
    py = float(bullseye_pixel[1])
    plane_x = (px - width / 2.0) / (width / 2.0)
    plane_y = (height / 2.0 - py) / (height / 2.0)
    return plane_x, plane_y


class OpenLoopAimController(AimController):
    def reset(self) -> None:
        pass

    def update(
        self,
        info: dict[str, Any],
        frame_shape: tuple[int, int, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, OpenLoopMetrics] | None:
        bullseye_pixel = info.get("bullseye_pixel")
        if not isinstance(bullseye_pixel, list) or len(bullseye_pixel) != 2:
            return None

        width = int(info.get("width", frame_shape[1]))
        height = int(info.get("height", frame_shape[0]))
        camera_fovy_deg = float(info["camera_fovy_deg"])
        camera_fovx_deg = float(info["camera_fovx_deg"])
        plane_x, plane_y = normalize_plane_coordinate(bullseye_pixel, width=width, height=height)

        spherical = backproject_to_spherical(
            (plane_x, plane_y),
            camera_fovy_deg=camera_fovy_deg,
            camera_fovx_deg=camera_fovx_deg,
        )

        action = np.zeros(6, dtype=np.float32)
        action[3] = np.clip(spherical.azimuth_rad / YAW_STEP_RAD, -1.0, 1.0)
        action[4] = np.clip(spherical.elevation_rad / PITCH_STEP_RAD, -1.0, 1.0)
        return action, OpenLoopMetrics(
            plane_x=plane_x,
            plane_y=plane_y,
            azimuth_deg=spherical.azimuth_deg,
            elevation_deg=spherical.elevation_deg,
        )
