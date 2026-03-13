from __future__ import annotations

from dataclasses import dataclass
from math import atan, atan2, degrees, radians, sqrt, tan
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class SphericalDirection:
    """Direction in a forward-facing spherical coordinate system.

    The returned unit vector uses the `(forward, right, up)` axis order:
    - `forward`: camera optical axis, positive straight ahead
    - `right`: positive to the image right
    - `up`: positive to the image top
    """

    vector: np.ndarray
    azimuth_rad: float
    elevation_rad: float

    @property
    def azimuth_deg(self) -> float:
        return degrees(self.azimuth_rad)

    @property
    def elevation_deg(self) -> float:
        return degrees(self.elevation_rad)


def _parse_plane_coordinate(target_plane_xy: Iterable[float]) -> tuple[float, float]:
    plane = tuple(float(v) for v in target_plane_xy)
    if len(plane) != 2:
        raise ValueError("target_plane_xy must contain exactly two values: (x, y)")
    return plane[0], plane[1]


def _resolve_horizontal_fov(camera_fovy_deg: float, camera_fovx_deg: float | None, aspect_ratio: float | None) -> float:
    if camera_fovx_deg is not None:
        return float(camera_fovx_deg)
    if aspect_ratio is None:
        raise ValueError("aspect_ratio is required when camera_fovx_deg is not provided")
    if aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive")
    fovy_rad = radians(float(camera_fovy_deg))
    fovx_rad = 2.0 * atan(aspect_ratio * tan(fovy_rad / 2.0))
    return degrees(fovx_rad)


def backproject_direction(
    target_plane_xy: Iterable[float],
    camera_fovy_deg: float,
    camera_fovx_deg: float | None = None,
    *,
    aspect_ratio: float | None = None,
    image_y_down: bool = False,
) -> np.ndarray:
    """Back-project a normalized image-plane point to a unit direction vector.

    Args:
        target_plane_xy:
            Normalized image-plane coordinates `(x, y)`, where the image center is
            `(0, 0)` and the view borders are approximately `[-1, 1]`.
        camera_fovy_deg:
            Vertical field of view in degrees.
        camera_fovx_deg:
            Horizontal field of view in degrees. If omitted, it is derived from
            `camera_fovy_deg` and `aspect_ratio`.
        aspect_ratio:
            `width / height`, used only when `camera_fovx_deg` is omitted.
        image_y_down:
            Set to `True` if the input plane coordinates use image convention
            (positive y downward). The output vector always uses positive up.

    Returns:
        A unit vector in `(forward, right, up)` order.
    """

    plane_x, plane_y = _parse_plane_coordinate(target_plane_xy)
    if image_y_down:
        plane_y = -plane_y

    fovx_deg = _resolve_horizontal_fov(camera_fovy_deg, camera_fovx_deg, aspect_ratio)
    tan_half_fovx = tan(radians(fovx_deg) / 2.0)
    tan_half_fovy = tan(radians(float(camera_fovy_deg)) / 2.0)

    forward = 1.0
    right = plane_x * tan_half_fovx
    up = plane_y * tan_half_fovy

    norm = sqrt(forward * forward + right * right + up * up)
    if norm == 0.0:
        raise ValueError("back-projected direction has zero norm")

    return np.array([forward / norm, right / norm, up / norm], dtype=np.float64)


def direction_to_spherical(direction: Iterable[float]) -> SphericalDirection:
    """Convert a direction vector to spherical angles around the forward axis.

    The azimuth angle is positive to the right, and the elevation angle is
    positive upward. Both are measured relative to the straight-ahead direction.
    """

    vector = np.asarray(tuple(float(v) for v in direction), dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("direction must contain exactly three values")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")

    unit = vector / norm
    forward, right, up = unit.tolist()
    azimuth_rad = atan2(right, forward)
    elevation_rad = atan2(up, sqrt(forward * forward + right * right))
    return SphericalDirection(vector=unit, azimuth_rad=azimuth_rad, elevation_rad=elevation_rad)


def backproject_to_spherical(
    target_plane_xy: Iterable[float],
    camera_fovy_deg: float,
    camera_fovx_deg: float | None = None,
    *,
    aspect_ratio: float | None = None,
    image_y_down: bool = False,
) -> SphericalDirection:
    """Back-project a normalized plane point and return spherical direction data."""

    direction = backproject_direction(
        target_plane_xy=target_plane_xy,
        camera_fovy_deg=camera_fovy_deg,
        camera_fovx_deg=camera_fovx_deg,
        aspect_ratio=aspect_ratio,
        image_y_down=image_y_down,
    )
    return direction_to_spherical(direction)


__all__ = [
    "SphericalDirection",
    "backproject_direction",
    "backproject_to_spherical",
    "direction_to_spherical",
]
