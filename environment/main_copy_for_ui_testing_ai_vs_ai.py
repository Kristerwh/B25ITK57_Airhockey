import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mujoco
import mujoco.viewer
import os
import numpy as np
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from rule_based_ai_agent_v31 import AI_script_v31 as script

env = AirHockeyBase()

env_info = env.env_info

n_agents = env_info.get("n_agents", 2)

env_info["actuator_joint_ids"] = env_info.get("actuator_joint_ids", [6, 7, 8, 9, 10, 11, 12])
env_info["_timestep"] = env_info.get("dt", 0.02)

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "env_settings", "environments", "data", "table.xml")

model = env._model
data = env._data

scene = mujoco.MjvScene(model, maxgeom=1000)
perturb = mujoco.MjvPerturb()
mujoco.mjv_defaultPerturb(perturb)

camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)

option = mujoco.MjvOption()
option.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

p_gain, d_gain, i_gain = 10.0, 1.0, 0.1
controller = MalletControl(env_info=env_info, debug=True)  # Enable debug if needed

paddle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_left")
paddle_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

scripted_ai = script.startup()
scripted_ai2 = script.startup()

#removing side panels/settings in the rendering window and then autoresizing window to fullscreen with pyautogui
#bcuz mujoco 3.xxx dosent support fullscreen for some reason
with mujoco.viewer.launch_passive(
    model, data,
    show_left_ui=False,
    show_right_ui=False
) as viewer:
    import pyautogui, time
    time.sleep(1)
    pyautogui.getWindowsWithTitle("MuJoCo")[0].maximize()

    while viewer.is_running():
        base_pos = np.array([data.qpos[5], data.qpos[6]])
        puck_pos = float(data.xpos[puck_id][0] * 1000) + 974, float(data.xpos[puck_id][1] * 1000) + 519
        puck_pos_reverted = 2 * 974 - (float(data.xpos[puck_id][0] * 1000) + 974), 2 * 519 - (float(data.xpos[puck_id][1] * 1000) + 519)
        # print(f"Puck: {puck_pos}")
        mallet_pos_script_ai = float(data.xpos[paddle_id][0] * 1000) + 974, float(data.xpos[paddle_id][1] * 1000) + 519
        mallet2_pos_script_ai = 2 * 974 - (float(data.xpos[paddle_id2][0] * 1000) + 974), 2 * 519 - (float(data.xpos[paddle_id2][1] * 1000) + 519)
        # print(f"Mallet: {mallet_pos_script_ai}")
        # print(f"Mallet: {mallet2_pos_script_ai}")
        puck_vel = np.array([data.qvel[0], data.qvel[1]])

        # print(mallet_pos_script_ai)

        ai_velocity = script.run(scripted_ai, puck_pos, mallet_pos_script_ai)
        ai_velocity2 = script.run(scripted_ai2, puck_pos_reverted, mallet2_pos_script_ai)
        ai_velocity2 = np.array([-ai_velocity2[0], -ai_velocity2[1]])
        action = np.concatenate((ai_velocity, ai_velocity2))  # Shape (4,)
        #print(ai_velocity)
        #print("test")
        print("-"*50)

        control_action = controller.apply_action(action)
        data.ctrl[:2] = control_action[:2]
        data.ctrl[2:4] = control_action[2:4]

        mujoco.mj_step(model, data)
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewer.sync()
