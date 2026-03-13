import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer

xml_path = Path("xml-models") / "turret.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 点动参数
JOG_CTRL = 0.6          # 点动时施加的控制量
JOG_DURATION = 0.12     # 每次点动持续时间，单位秒

# 当前点动命令的剩余时间
yaw_cmd = 0.0
pitch_cmd = 0.0
yaw_time_left = 0.0
pitch_time_left = 0.0


def key_callback(keycode: int) -> None:
    """
    键盘点动：
    J -> yaw 正向点动
    L -> yaw 负向点动
    I -> pitch 正向点动
    K -> pitch 负向点动
    """
    global yaw_cmd, pitch_cmd, yaw_time_left, pitch_time_left

    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    if key == "j":
        yaw_cmd = +JOG_CTRL
        yaw_time_left = JOG_DURATION
    elif key == "l":
        yaw_cmd = -JOG_CTRL
        yaw_time_left = JOG_DURATION
    elif key == "i":
        pitch_cmd = +JOG_CTRL
        pitch_time_left = JOG_DURATION
    elif key == "k":
        pitch_cmd = -JOG_CTRL
        pitch_time_left = JOG_DURATION


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    dt = model.opt.timestep

    while viewer.is_running():
        # yaw 点动脉冲
        if yaw_time_left > 0:
            data.ctrl[0] = yaw_cmd
            yaw_time_left -= dt
        else:
            data.ctrl[0] = 0.0

        # pitch 点动脉冲
        if pitch_time_left > 0:
            data.ctrl[1] = pitch_cmd
            pitch_time_left -= dt
        else:
            data.ctrl[1] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

print("yaw =", data.qpos[0], "pitch =", data.qpos[1])