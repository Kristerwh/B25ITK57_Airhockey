import sys
import os
import time

# import keyboard
import numpy as np
import pyautogui
import mujoco
import mujoco.viewer

from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from rule_based_ai_agent_v31 import AI_script_v31 as script

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

env = AirHockeyBase(n_agents=1)
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

def disable_left_mouse(viewer):
    def blocked_mouse_handler(*args, **kwargs):
        pass

    viewer._render_mouse = blocked_mouse_handler

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

# from pynput import mouse, keyboard
# def on_click(x, y, button, pressed):
#     if button == mouse.Button.left:
#         return False  # Block the click event
# def esc_listener():
#     def on_press(key):
#         if key == keyboard.Key.esc:
#             print("Escape pressed. Exiting...")
#             sys.exit()
#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()
#
# # Start the listener in the background
# mouse_listener = mouse.Listener(on_click=on_click, suppress=True)
# mouse_listener.start()
# esc_listener()

from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from environment.PPO_training.ppo_trainer import PPOTrainer
ppo_agent = PPOTrainer(obs_dim=8, action_dim=2)
ppo_agent.load("../environment/PPO_training/PPO_training_saved_models/saved_model")

from collections import deque
sequence_length = 10
obs_sequence = deque(maxlen=sequence_length)

def strip_z(obs):
    puck_pos = env.obs_helper.get_from_obs(obs, "puck_pos")[:2]
    puck_vel = np.concatenate([
        env.obs_helper.get_from_obs(obs, "puck_x_vel"),
        env.obs_helper.get_from_obs(obs, "puck_y_vel")
    ])
    mallet_pos = env.obs_helper.get_from_obs(obs, "paddle_left_pos")[:2]
    mallet_vel = np.concatenate([
        env.obs_helper.get_from_obs(obs, "paddle_left_x_vel"),
        env.obs_helper.get_from_obs(obs, "paddle_left_y_vel")
    ])
    return np.concatenate([puck_pos, puck_vel, mallet_pos, mallet_vel])

counter = 0


with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    time.sleep(1)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -90
    viewer.cam.distance = 1.39
    viewer.cam.trackbodyid = -1
    viewer.cam.fixedcamid = -1
    # obs_sequence = deque(maxlen=sequence_length)

    # if counter % 2 == 0:
    #     obs_raw = strip_z(env.reset("opponent_y"))
    #     obs_normalized = strip_z(env.reset("opponent_y"))
    # else:
    #     obs_raw = strip_z(env.reset("y"))
    #     obs_normalized = strip_z(env.reset("y"))
    # for _ in range(sequence_length):
    #     obs_sequence.append(obs_normalized)
    # obs = np.array(obs_sequence)
    obs = env.reset(("y"))
    obs = strip_z(obs)

    if abs(viewer.cam.azimuth - 90) > 1 or abs(viewer.cam.elevation + 90) > 1:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -90
    pyautogui.getWindowsWithTitle("MuJoCo")[0].maximize()
    while viewer.is_running():
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        # viewer.cam.azimuth = 90
        # viewer.cam.elevation = -90
        # viewer.cam.distance = 1.39
        # viewer.cam.trackbodyid = -1
        # viewer.cam.fixedcamid = -1
        #
        # if abs(viewer.cam.azimuth - 90) > 1 or abs(viewer.cam.elevation + 90) > 1:
        #     viewer.cam.azimuth = 90
        #     viewer.cam.elevation = -90

        mouse_x, mouse_y = pyautogui.position()
        target_x = (mouse_x - 974) / 1000.0
        target_y = (2 * 519 - mouse_y) / 1000.0
        target_pos = np.clip(np.array([target_x, target_y]), [-0.95, -0.45], [0.95, 0.45])
        target_pos = np.array([((mouse_x/965)-1.795), -((mouse_y/970) -0.55)])

        data.qpos[5:7] = target_pos
        data.qvel[5:7] = 0

        #puck_pos = float(data.xpos[puck_id][0]) * 1000 + 974, float(data.xpos[puck_id][1]) * 1000 + 519
        #puck_pos_reverted = 2 * 974 - puck_pos[0], 2 * 519 - puck_pos[1]
        #mallet2_pos = 2 * 974 - (float(data.xpos[paddle_id2][0]) * 1000 + 974), \
                      #2 * 519 - (float(data.xpos[paddle_id2][1]) * 1000 + 519)

        action, _ = ppo_agent.act(strip_z(env._obs))
        #actionsend = np.concatenate((action, target_pos))
        #print(action, target_pos)
        #print(actionsend)
        next_obs, reward, absorbing, _ = env.step(action)
        #Rule-based AI movement
        #ai_velocity2 = np.array(script.run(scripted_ai2, puck_pos_reverted, mallet2_pos))
        #ai_velocity2 = -ai_velocity2

        #action2 = np.concatenate(([0, 0], action))
        #control_action = controller.apply_action(action2)

        #data.ctrl[2:4] = control_action[2:4]

        #mujoco.mj_step(model, data)

        # obs = env._create_observation(env.obs_helper._build_obs(data))
        if absorbing:
            print("Goal detected!")
            obs = strip_z(env.reset("y"))

        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewer.sync()
