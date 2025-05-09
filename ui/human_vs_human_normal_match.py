import time
import numpy as np
import pyautogui
import mujoco
import mujoco.viewer

from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase

env = AirHockeyBase(n_agents=2)
model, data = env._model, env._data
controller = MalletControl(env_info=env.env_info, debug=False)

for side in ["paddle_left", "paddle_right"]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side)
    model.body_pos[body_id][2] = 0.015

prev_pos_left = np.array([-0.6, 0.0])
prev_pos_right = np.array([0.6, 0.0])

def reset_game():
    global prev_pos_left, prev_pos_right
    env.reset()
    #Left paddle
    data.qpos[3:5] = [-0.1, 0.0]
    #Right paddle
    data.qpos[5:7] = [0.1, 0.0]
    data.qvel[3:7] = 0
    mujoco.mj_forward(model, data)
    prev_pos_left = np.array([-0.6, 0.0])
    prev_pos_right = np.array([0.6, 0.0])

reset_game()

scene, camera, option = mujoco.MjvScene(model, 1000), mujoco.MjvCamera(), mujoco.MjvOption()
mujoco.mjv_defaultCamera(camera)
mujoco.mjv_defaultOption(option)

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    pyautogui.getWindowsWithTitle("MuJoCo")[0].maximize()

    while viewer.is_running():
        #left = player 1, right = player 2
        mouse_x, mouse_y = pyautogui.position()

        if mouse_x < 974:
            target_x = (mouse_x - 487) / 1000.0
            target_y = (2 * 519 - mouse_y) / 1000.0
            target_pos = np.clip([target_x, target_y], [-0.95, -0.45], [0.0, 0.45])
            velocity = np.clip((target_pos - prev_pos_left) / env.dt, -0.3, 0.3)
            prev_pos_left = target_pos
            data.qpos[3:5] = target_pos
            data.qvel[3:5] = velocity

        else:
            target_x = (mouse_x - 974) / 1000.0
            target_y = (2 * 519 - mouse_y) / 1000.0
            target_pos = np.clip([target_x, target_y], [0.0, -0.45], [0.95, 0.45])
            velocity = np.clip((target_pos - prev_pos_right) / env.dt, -0.3, 0.3)
            prev_pos_right = target_pos
            data.qpos[5:7] = target_pos
            data.qvel[5:7] = velocity

        mujoco.mj_step(model, data)

        obs = env._create_observation(env.obs_helper._build_obs(data))
        if AirHockeyBase.is_absorbing_ui(obs):
            print("Goal scored — resetting!")
            reset_game()

        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewer.sync()
