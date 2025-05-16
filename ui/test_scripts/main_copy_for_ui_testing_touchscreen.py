import sys
import os
import time
import numpy as np
import pyautogui
import mujoco
import mujoco.viewer

from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from rule_based_ai_agent_v31 import AI_script_v31 as script

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

env = AirHockeyBase()
model = env._model
data = env._data
controller = MalletControl(env_info=env.env_info, debug=True)

left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_left")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
model.body_pos[left_id][2] = 0.015
model.body_pos[right_id][2] = 0.015

paddle_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

scripted_ai2 = script.startup()

def reset_game():
    data.qpos[0:2] = np.random.uniform(-0.4, -0.2), np.random.uniform(-0.2, 0.2)
    data.qvel[0:2] = 0
    data.qpos[3:5] = [-0.1, 0.0]
    data.qpos[5:7] = [0.1, 0.0]
    data.qvel[3:7] = 0
    mujoco.mj_forward(model, data)

reset_game()

scene = mujoco.MjvScene(model, maxgeom=1000)
camera = mujoco.MjvCamera()
# camera.type = mujoco.mjtCamera.mjCAMERA_FREE
# camera.lookat = np.array([0.0, 0.0, 0.0])     #sentrer kamera i midten
# camera.azimuth = 90                           #horizontal = 0, vertical = 90
# camera.elevation = -90                        #ser rett ned
# camera.distance = 1.3                         #zoom in og zoom out, 1.2/1.3 funker
# camera.trackbodyid = -1                       #jager ikke bodys(xml objekter altså)
# camera.fixedcamid = -1
option = mujoco.MjvOption()
mujoco.mjv_defaultOption(option)
option.frame = mujoco.mjtFrame.mjFRAME_NONE
#mujoco.mjv_defaultCamera(camera)

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    time.sleep(1)
    pyautogui.getWindowsWithTitle("MuJoCo")[0].maximize()
    while viewer.is_running():

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -90
        viewer.cam.distance = 1.39
        viewer.cam.trackbodyid = -1
        viewer.cam.fixedcamid = -1

        if abs(viewer.cam.azimuth - 90) > 1 or abs(viewer.cam.elevation + 90) > 1:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -90

        mouse_x, mouse_y = pyautogui.position()
        target_x = (mouse_x - 974) / 1000.0
        target_y = (2 * 519 - mouse_y) / 1000.0
        target_pos = np.clip(np.array([target_x, target_y]), [-0.95, -0.45], [0.95, 0.45])

        data.qpos[3:5] = target_pos
        data.qvel[3:5] = 0

        puck_pos = float(data.xpos[puck_id][0]) * 1000 + 974, float(data.xpos[puck_id][1]) * 1000 + 519
        puck_pos_reverted = 2 * 974 - puck_pos[0], 2 * 519 - puck_pos[1]
        mallet2_pos = 2 * 974 - (float(data.xpos[paddle_id2][0]) * 1000 + 974), \
                      2 * 519 - (float(data.xpos[paddle_id2][1]) * 1000 + 519)

        # Rule-based AI movement
        ai_velocity2 = np.array(script.run(scripted_ai2, puck_pos_reverted, mallet2_pos))
        ai_velocity2 = -ai_velocity2

        action = np.concatenate(([0, 0], ai_velocity2))
        control_action = controller.apply_action(action)

        data.ctrl[2:4] = control_action[2:4]

        mujoco.mj_step(model, data)

        obs = env._create_observation(env.obs_helper._build_obs(data))
        if AirHockeyBase.is_absorbing_ui(obs):
            print("Goal detected!")
            reset_game()

        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewer.sync()
