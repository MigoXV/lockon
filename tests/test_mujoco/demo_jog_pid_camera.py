import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

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

# 平移速度控制（持续按键时的速度）
MOVE_STEP = 0.03
x_target = 0.0
y_target = 0.0

# 离屏渲染尺寸
CAM_WIDTH = 320
CAM_HEIGHT = 240


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def process_key(key_code: int) -> None:
    """通过 OpenCV 窗口处理按键，避免与 MuJoCo viewer 快捷键冲突。"""
    global yaw_target, pitch_target, x_target, y_target

    ch = chr(key_code & 0xFF)

    # 炮台旋转 IJKL
    if ch == "j":
        yaw_target += JOG_ANGLE
    elif ch == "l":
        yaw_target -= JOG_ANGLE
    elif ch == "i":
        pitch_target += JOG_ANGLE
    elif ch == "k":
        pitch_target -= JOG_ANGLE
    # 底座平移 WASD
    elif ch == "w":
        x_target += MOVE_STEP
    elif ch == "s":
        x_target -= MOVE_STEP
    elif ch == "a":
        y_target += MOVE_STEP
    elif ch == "d":
        y_target -= MOVE_STEP
    else:
        return

    # 按 joint range 限制目标
    yaw_target = clamp(yaw_target, -3.1416, 3.1416)
    pitch_target = clamp(pitch_target, 0.0, 1.5708)
    x_target = clamp(x_target, -1.5, 1.5)
    y_target = clamp(y_target, -1.5, 1.5)


# 离屏渲染器
renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

# 获取 turret_cam 的 id
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "turret_cam")

cv2.namedWindow("Turret Camera", cv2.WINDOW_NORMAL)

with mujoco.viewer.launch_passive(model, data) as viewer:
    dt = model.opt.timestep
    frame_skip = 5  # 每 5 步渲染一帧相机画面
    step_count = 0

    while viewer.is_running():
        # qpos: [slide_x, slide_y, yaw, pitch]
        sx = data.qpos[0]
        sy = data.qpos[1]
        yaw = data.qpos[2]
        pitch = data.qpos[3]
        sx_vel = data.qvel[0]
        sy_vel = data.qvel[1]
        yaw_vel = data.qvel[2]
        pitch_vel = data.qvel[3]

        # PD 控制 - 平移
        KP_move = 20.0
        KD_move = 5.0
        sx_ctrl = KP_move * (x_target - sx) - KD_move * sx_vel
        sy_ctrl = KP_move * (y_target - sy) - KD_move * sy_vel

        # PD 控制 - 旋转
        yaw_ctrl = KP * (yaw_target - yaw) - KD * yaw_vel
        pitch_ctrl = KP * (pitch_target - pitch) - KD * pitch_vel

        # ctrl: [slide_x, slide_y, yaw, pitch]
        data.ctrl[0] = clamp(sx_ctrl, -1.0, 1.0)
        data.ctrl[1] = clamp(sy_ctrl, -1.0, 1.0)
        data.ctrl[2] = clamp(yaw_ctrl, -1.0, 1.0)
        data.ctrl[3] = clamp(pitch_ctrl, -1.0, 1.0)

        mujoco.mj_step(model, data)
        viewer.sync()

        # 按 frame_skip 频率更新相机画面
        step_count += 1
        if step_count % frame_skip == 0:
            renderer.update_scene(data, camera=cam_id)
            img = renderer.render()
            # MuJoCo 输出 RGB，OpenCV 需要 BGR
            cv2.imshow("Turret Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            if key != -1:
                process_key(key)

        time.sleep(dt)

renderer.close()
cv2.destroyAllWindows()
print(f"x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} yaw={data.qpos[2]:.3f} pitch={data.qpos[3]:.3f}")
