import os
import time
import mujoco
import mujoco.viewer

xml_path = os.path.join("xml-models/turret.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = 0.5
        data.ctrl[1] = -0.2

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print("yaw =", data.qpos[0], "pitch =", data.qpos[1])
