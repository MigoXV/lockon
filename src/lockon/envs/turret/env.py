from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import gymnasium as gym

if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from gymnasium import spaces


class TurretEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    RING_BOUNDARIES = np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13], dtype=np.float64)
    RING_NAMES = ("靶心(10环)", "9环", "8环", "7环", "6环", "5环", "4环")
    TARGET_STEP_SCALE = np.array([0.03, 0.03, 0.05, 0.08, 0.08], dtype=np.float64)

    def __init__(
        self,
        xml_path: str | Path | None = None,
        render_mode: str | None = "rgb_array",
        camera_name: str = "turret_cam",
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fovy_deg: float | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.xml_path = self._resolve_xml_path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.camera_name = camera_name
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.dt = float(self.model.opt.timestep)

        self.qpos_size = 5
        self.qvel_size = 5
        self._initial_qpos = self.data.qpos[: self.qpos_size].copy()
        self.targets = self._initial_qpos.copy()

        self._position_low = self.model.jnt_range[: self.qpos_size, 0].astype(np.float32)
        self._position_high = self.model.jnt_range[: self.qpos_size, 1].astype(np.float32)

        obs_low = np.concatenate(
            [
                self._position_low,
                np.full(self.qvel_size, -np.inf, dtype=np.float32),
                self._position_low,
            ]
        )
        obs_high = np.concatenate(
            [
                self._position_high,
                np.full(self.qvel_size, np.inf, dtype=np.float32),
                self._position_high,
            ]
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(self.qpos_size, dtype=np.float32),
            high=np.ones(self.qpos_size, dtype=np.float32),
            dtype=np.float32,
        )

        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.bullseye_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "bullseye_center")
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

        if camera_fovy_deg is not None:
            self.model.cam_fovy[self.cam_id] = float(camera_fovy_deg)

        self.camera_fovy_rad = np.radians(float(self.model.cam_fovy[self.cam_id]))
        self.camera_fovx_rad = 2.0 * np.arctan(
            (self.camera_width / self.camera_height) * np.tan(self.camera_fovy_rad / 2.0)
        )

        self.renderer: mujoco.Renderer | None = None

    @staticmethod
    def _resolve_xml_path(xml_path: str | Path | None) -> Path:
        if xml_path is not None:
            return Path(xml_path)
        return Path(__file__).resolve().parents[4] / "xml-models" / "turret.xml"

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate(
            [
                self.data.qpos[: self.qpos_size],
                self.data.qvel[: self.qvel_size],
                self.targets,
            ]
        )
        return obs.astype(np.float32)

    def _clamp_targets(self) -> None:
        self.targets = np.clip(self.targets, self._position_low, self._position_high)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.qpos_size,):
            raise ValueError(f"expected action shape {(self.qpos_size,)}, got {action.shape}")
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _apply_pd_control(self) -> None:
        qpos = self.data.qpos[: self.qpos_size]
        qvel = self.data.qvel[: self.qvel_size]

        kp = np.array([20.0, 20.0, 6.0, 8.0, 8.0], dtype=np.float64)
        kd = np.array([5.0, 5.0, 1.2, 1.5, 1.5], dtype=np.float64)
        ctrl = kp * (self.targets - qpos) - kd * qvel
        self.data.ctrl[: self.qpos_size] = np.clip(ctrl, -1.0, 1.0)

    def _aim_error(self) -> float:
        tip_pos = self.data.site_xpos[self.tip_site_id]
        tip_xmat = self.data.site_xmat[self.tip_site_id].reshape(3, 3)
        ray_dir = tip_xmat[:, 0]
        target_pos = self.data.site_xpos[self.bullseye_site_id]
        plane_normal = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        denom = float(np.dot(ray_dir, plane_normal))
        if abs(denom) < 1e-9:
            return float("inf")

        t = float(np.dot(target_pos - tip_pos, plane_normal) / denom)
        if t < 0:
            return float("inf")

        hit_world = tip_pos + t * ray_dir
        offset = hit_world - target_pos
        return float(np.linalg.norm(offset[1:]))

    def get_state_info(self) -> dict[str, Any]:
        return {
            "qpos": self.data.qpos[: self.qpos_size].astype(np.float32).copy(),
            "qvel": self.data.qvel[: self.qvel_size].astype(np.float32).copy(),
            "targets": self.targets.astype(np.float32).copy(),
            "aim_error": self._aim_error(),
            "camera_fovy_deg": float(np.degrees(self.camera_fovy_rad)),
            "camera_fovx_deg": float(np.degrees(self.camera_fovx_rad)),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self.qpos_size] = self._initial_qpos
        self.data.qvel[: self.qvel_size] = 0.0

        if options and "targets" in options:
            self.targets = np.asarray(options["targets"], dtype=np.float64).copy()
        else:
            self.targets = self.data.qpos[: self.qpos_size].copy()

        self._clamp_targets()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self.get_state_info()

    def step_control(self, action: np.ndarray) -> tuple[float, bool, bool, dict[str, Any]]:
        clipped_action = self._normalize_action(action)
        self.targets = self.targets + clipped_action.astype(np.float64) * self.TARGET_STEP_SCALE
        self._clamp_targets()
        self._apply_pd_control()
        mujoco.mj_step(self.model, self.data)

        info = self.get_state_info()
        reward = -info["aim_error"] if np.isfinite(info["aim_error"]) else -10.0
        return reward, False, False, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward, terminated, truncated, info = self.step_control(action)
        return self._get_obs(), reward, terminated, truncated, info

    def world_to_pixel(self, point_world: np.ndarray) -> tuple[int, int] | None:
        cam_pos = self.data.cam_xpos[self.cam_id].copy()
        cam_mat = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        local = cam_mat.T @ (point_world - cam_pos)

        if local[2] >= 0:
            return None

        focal = (self.camera_height / 2.0) / np.tan(self.camera_fovy_rad / 2.0)
        px = int(self.camera_width / 2.0 - focal * local[0] / local[2])
        py = int(self.camera_height / 2.0 + focal * local[1] / local[2])

        if 0 <= px < self.camera_width and 0 <= py < self.camera_height:
            return px, py
        return None

    def fire(self) -> dict[str, Any]:
        tip_pos = self.data.site_xpos[self.tip_site_id].copy()
        tip_xmat = self.data.site_xmat[self.tip_site_id].reshape(3, 3)
        ray_dir = tip_xmat[:, 0]
        target_pos = self.data.site_xpos[self.bullseye_site_id].copy()
        plane_normal = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        denom = float(np.dot(ray_dir, plane_normal))
        if abs(denom) < 1e-9:
            return {"hit": False, "reason": "射线与靶面平行，未命中"}

        t = float(np.dot(target_pos - tip_pos, plane_normal) / denom)
        if t < 0:
            return {"hit": False, "reason": "靶面在身后，未命中"}

        hit_world = tip_pos + t * ray_dir
        offset = hit_world - target_pos
        dy = float(offset[1])
        dz = float(offset[2])
        radius = float(np.sqrt(dy**2 + dz**2))
        theta_deg = float(np.degrees(np.arctan2(dz, dy)))

        zone = "脱靶"
        for idx, boundary in enumerate(self.RING_BOUNDARIES):
            if radius <= boundary:
                zone = self.RING_NAMES[idx]
                break

        return {
            "hit": True,
            "radius": radius,
            "theta_deg": theta_deg,
            "dy": dy,
            "dz": dz,
            "zone": zone,
            "hit_world": hit_world.copy(),
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)

        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
