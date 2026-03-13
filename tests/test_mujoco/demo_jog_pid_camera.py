import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

xml_path = Path("xml-models") / "turret.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# site id
tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
bullseye_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bullseye_center")

# 靶的环界定义（从内到外）
RING_BOUNDARIES = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13]
RING_NAMES = ["靶心(10环)", "9环", "8环", "7环", "6环", "5环", "4环"]

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

# 相机 fovy（与 XML 一致）
CAM_FOVY_RAD = np.radians(60)


def world_to_pixel(point_world: np.ndarray) -> tuple[int, int] | None:
    """将世界坐标投影到 turret_cam 的像素坐标，返回 (px, py) 或 None（在视野外/身后）。"""
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    # 相机局部坐标系：MuJoCo camera frame → x=right, y=up, z=-forward
    d = point_world - cam_pos
    local = cam_mat.T @ d  # 转到相机坐标系

    # z 分量 < 0 表示在相机前方（MuJoCo 相机看向 -z）
    if local[2] >= 0:
        return None  # 在相机身后

    # 透视投影
    f = (CAM_HEIGHT / 2.0) / np.tan(CAM_FOVY_RAD / 2.0)
    px = int(CAM_WIDTH / 2.0 - f * local[0] / local[2])
    py = int(CAM_HEIGHT / 2.0 + f * local[1] / local[2])

    if 0 <= px < CAM_WIDTH and 0 <= py < CAM_HEIGHT:
        return (px, py)
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fire() -> None:
    """从 tip 沿炮管方向发射射线，检测与靶面的交点。"""
    tip_pos = data.site_xpos[tip_site_id].copy()
    tip_xmat = data.site_xmat[tip_site_id].reshape(3, 3)
    ray_dir = tip_xmat[:, 0]  # site frame 的 x 轴 = 炮管前方

    target_pos = data.site_xpos[bullseye_site_id].copy()

    # 靶面法线：靶子朝 -x 方向（面向炮台）
    plane_normal = np.array([-1.0, 0.0, 0.0])

    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-9:
        print("[FIRE] 射线与靶面平行，未命中")
        return

    t = np.dot(target_pos - tip_pos, plane_normal) / denom
    if t < 0:
        print("[FIRE] 靶面在身后，未命中")
        return

    hit_world = tip_pos + t * ray_dir
    offset = hit_world - target_pos

    # 靶面上 y/z 偏移 → 极坐标
    dy = offset[1]
    dz = offset[2]
    r = np.sqrt(dy**2 + dz**2)
    theta_deg = np.degrees(np.arctan2(dz, dy))

    # 判定命中区域
    zone = "脱靶"
    for i, boundary in enumerate(RING_BOUNDARIES):
        if r <= boundary:
            zone = RING_NAMES[i]
            break

    print(f"[FIRE] 命中! 极坐标: r={r:.4f}m, θ={theta_deg:.1f}°  |  偏移: dy={dy:.4f} dz={dz:.4f}  |  {zone}")


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
    # 开火 F
    elif ch == "f":
        fire()
        return
    else:
        return

    # 按 joint range 限制目标
    yaw_target = clamp(yaw_target, -3.1416, 3.1416)
    pitch_target = clamp(pitch_target, 0.0, 1.5708)
    x_target = clamp(x_target, -0.8, 0.15)
    y_target = clamp(y_target, -0.75, 0.75)


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
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 在画面上标注靶心蓝点
            bullseye_world = data.site_xpos[bullseye_site_id].copy()
            px_pos = world_to_pixel(bullseye_world)
            if px_pos is not None:
                cv2.circle(frame, px_pos, 5, (255, 0, 0), -1)  # BGR 蓝色实心圆

            # 中心十字准星（绿色）
            cx, cy = CAM_WIDTH // 2, CAM_HEIGHT // 2
            cross_size = 10
            cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (0, 255, 0), 1)
            cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (0, 255, 0), 1)

            cv2.imshow("Turret Camera", frame)
            key = cv2.waitKey(1)
            if key != -1:
                process_key(key)

        time.sleep(dt)

renderer.close()
cv2.destroyAllWindows()
print(f"x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} yaw={data.qpos[2]:.3f} pitch={data.qpos[3]:.3f}")
