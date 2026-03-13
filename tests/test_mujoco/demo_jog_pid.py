import time
from pathlib import Path

import mujoco
import mujoco.viewer

xml_path = Path("xml-models") / "turret.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 每次点动的角度（弧度）
JOG_ANGLE = 0.08

# 简单 PD 参数
KP = 8.0
KD = 1.5

yaw_target = 0.0
pitch_target = 0.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def key_callback(keycode: int) -> None:
    global yaw_target, pitch_target

    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    if key == "j":
        yaw_target += JOG_ANGLE
    elif key == "l":
        yaw_target -= JOG_ANGLE
    elif key == "i":
        pitch_target += JOG_ANGLE
    elif key == "k":
        pitch_target -= JOG_ANGLE

    # 按 joint range 限制目标
    yaw_target = clamp(yaw_target, -1.57, 1.57)
    pitch_target = clamp(pitch_target, -0.9, 0.9)


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    dt = model.opt.timestep

    while viewer.is_running():
        yaw = data.qpos[0]
        pitch = data.qpos[1]
        yaw_vel = data.qvel[0]
        pitch_vel = data.qvel[1]

        # PD 控制
        yaw_ctrl = KP * (yaw_target - yaw) - KD * yaw_vel
        pitch_ctrl = KP * (pitch_target - pitch) - KD * pitch_vel

        # 你的 motor ctrlrange 是 [-1, 1]
        data.ctrl[0] = clamp(yaw_ctrl, -1.0, 1.0)
        data.ctrl[1] = clamp(pitch_ctrl, -1.0, 1.0)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

print("yaw =", data.qpos[0], "pitch =", data.qpos[1])
