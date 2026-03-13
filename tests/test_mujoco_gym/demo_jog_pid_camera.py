import time

import cv2
import numpy as np

from lockon.envs.turret import TurretEnv

env = TurretEnv()
env.reset()

CAM_WIDTH = env.camera_width
CAM_HEIGHT = env.camera_height


def process_key(key_code: int) -> np.ndarray:
    """通过 OpenCV 窗口处理按键，避免与 MuJoCo viewer 快捷键冲突。"""
    ch = chr(key_code & 0xFF)
    action = np.zeros(4, dtype=np.float32)

    if ch == "j":
        action[2] = 1.0
    elif ch == "l":
        action[2] = -1.0
    elif ch == "i":
        action[3] = 1.0
    elif ch == "k":
        action[3] = -1.0
    elif ch == "w":
        action[0] = 1.0
    elif ch == "s":
        action[0] = -1.0
    elif ch == "a":
        action[1] = 1.0
    elif ch == "d":
        action[1] = -1.0
    elif ch == "f":
        result = env.fire()
        if result["hit"]:
            print(
                f"[FIRE] 命中! 极坐标: r={result['radius']:.4f}m, "
                f"θ={result['theta_deg']:.1f}°  |  偏移: dy={result['dy']:.4f} "
                f"dz={result['dz']:.4f}  |  {result['zone']}"
            )
        else:
            print(f"[FIRE] {result['reason']}")

    return action


cv2.namedWindow("Turret Camera", cv2.WINDOW_NORMAL)

frame_skip = 5
step_count = 0

try:
    while True:
        env.step(np.zeros(4, dtype=np.float32))

        step_count += 1
        if step_count % frame_skip == 0:
            img = env.render()
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            bullseye_world = env.data.site_xpos[env.bullseye_site_id].copy()
            px_pos = env.world_to_pixel(bullseye_world)
            if px_pos is not None:
                cv2.circle(frame, px_pos, 5, (255, 0, 0), -1)

            cx, cy = CAM_WIDTH // 2, CAM_HEIGHT // 2
            cross_size = 10
            cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (0, 255, 0), 1)
            cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (0, 255, 0), 1)

            cv2.imshow("Turret Camera", frame)
            key = cv2.waitKey(1)
            if key != -1:
                if (key & 0xFF) == 27:
                    break
                env.step(process_key(key))

        time.sleep(env.dt)
finally:
    env.close()
    cv2.destroyAllWindows()

print(
    f"x={env.data.qpos[0]:.3f} y={env.data.qpos[1]:.3f} "
    f"yaw={env.data.qpos[2]:.3f} pitch={env.data.qpos[3]:.3f}"
)
