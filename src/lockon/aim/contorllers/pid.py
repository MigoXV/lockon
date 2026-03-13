from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lockon.aim.back_projection import backproject_to_spherical
from lockon.aim.contorllers.base import AimController, AimMetrics
from lockon.aim.contorllers.open_loop import PITCH_STEP_RAD, YAW_STEP_RAD, normalize_plane_coordinate


@dataclass
class AxisPid:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    output_limit: float
    deadband: float = 0.0

    integral: float = 0.0
    prev_error: float = 0.0
    initialized: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float) -> tuple[float, dict[str, float]]:
        if abs(error) < self.deadband:
            error = 0.0

        if not self.initialized:
            derivative = 0.0
            self.initialized = True
        else:
            derivative = (error - self.prev_error) / max(dt, 1e-6)

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = float(np.clip(output, -self.output_limit, self.output_limit))

        self.prev_error = error
        return output, {
            "p": self.kp * error,
            "i": self.ki * self.integral,
            "d": self.kd * derivative,
            "output": output,
        }


@dataclass(frozen=True, slots=True)
class PidAimConfig:
    ff_gain: float = 0.35
    yaw_kp: float = 1.1
    yaw_ki: float = 0.08
    yaw_kd: float = 0.18
    pitch_kp: float = 1.1
    pitch_ki: float = 0.08
    pitch_kd: float = 0.18
    pid_deadband: float = 0.002
    integral_limit: float = 0.4
    feedback_limit: float = 0.7


@dataclass(frozen=True, slots=True)
class PidAimMetrics(AimMetrics):
    plane_x: float
    plane_y: float
    azimuth_deg: float
    elevation_deg: float
    yaw_ff: float
    pitch_ff: float
    yaw_fb: float
    pitch_fb: float
    yaw_p: float
    yaw_i: float
    yaw_d: float
    pitch_p: float
    pitch_i: float
    pitch_d: float

    def as_dict(self) -> dict[str, float]:
        return {
            "plane_x": self.plane_x,
            "plane_y": self.plane_y,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "yaw_ff": self.yaw_ff,
            "pitch_ff": self.pitch_ff,
            "yaw_fb": self.yaw_fb,
            "pitch_fb": self.pitch_fb,
            "yaw_p": self.yaw_p,
            "yaw_i": self.yaw_i,
            "yaw_d": self.yaw_d,
            "pitch_p": self.pitch_p,
            "pitch_i": self.pitch_i,
            "pitch_d": self.pitch_d,
        }


class PidAimController(AimController):
    def __init__(self, config: PidAimConfig | None = None) -> None:
        self.config = config or PidAimConfig()
        self.yaw_pid = AxisPid(
            kp=self.config.yaw_kp,
            ki=self.config.yaw_ki,
            kd=self.config.yaw_kd,
            integral_limit=self.config.integral_limit,
            output_limit=self.config.feedback_limit,
            deadband=self.config.pid_deadband,
        )
        self.pitch_pid = AxisPid(
            kp=self.config.pitch_kp,
            ki=self.config.pitch_ki,
            kd=self.config.pitch_kd,
            integral_limit=self.config.integral_limit,
            output_limit=self.config.feedback_limit,
            deadband=self.config.pid_deadband,
        )

    def reset(self) -> None:
        self.yaw_pid.reset()
        self.pitch_pid.reset()

    def update(
        self,
        info: dict[str, Any],
        frame_shape: tuple[int, int, int],
        dt: float,
    ) -> tuple[np.ndarray, PidAimMetrics] | None:
        bullseye_pixel = info.get("bullseye_pixel")
        if not isinstance(bullseye_pixel, list) or len(bullseye_pixel) != 2:
            self.reset()
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

        yaw_ff = self.config.ff_gain * (spherical.azimuth_rad / YAW_STEP_RAD)
        pitch_ff = self.config.ff_gain * (spherical.elevation_rad / PITCH_STEP_RAD)
        yaw_fb, yaw_terms = self.yaw_pid.update(plane_x, dt)
        pitch_fb, pitch_terms = self.pitch_pid.update(plane_y, dt)

        action = np.zeros(6, dtype=np.float32)
        action[3] = np.clip(yaw_ff + yaw_fb, -1.0, 1.0)
        action[4] = np.clip(pitch_ff + pitch_fb, -1.0, 1.0)
        return action, PidAimMetrics(
            plane_x=plane_x,
            plane_y=plane_y,
            azimuth_deg=spherical.azimuth_deg,
            elevation_deg=spherical.elevation_deg,
            yaw_ff=yaw_ff,
            pitch_ff=pitch_ff,
            yaw_fb=yaw_fb,
            pitch_fb=pitch_fb,
            yaw_p=yaw_terms["p"],
            yaw_i=yaw_terms["i"],
            yaw_d=yaw_terms["d"],
            pitch_p=pitch_terms["p"],
            pitch_i=pitch_terms["i"],
            pitch_d=pitch_terms["d"],
        )
