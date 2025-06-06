import time
import numpy as np
import pyautogui
import mujoco
import mujoco.viewer

from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from environment.PPO_training.ppo_trainer import PPOTrainer

env = AirHockeyBase()
model, data = env._model, env._data
controller = MalletControl(env_info=env.env_info, debug=False)

#mallet heights to over glitches
left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_left")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
model.body_pos[left_id][2] = 0.015
model.body_pos[right_id][2] = 0.015

ppo_trainer = PPOTrainer(obs_dim=8, action_dim=2)
ppo_trainer.load("../environment/PPO_training/PPO_training_saved_models/saved_model")

prev_touch_pos = np.array([0.6, 0.0])

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

def reset_game():
    global prev_touch_pos
    env.reset()
    data.qpos[3:5] = [-0.1, 0.0]
    data.qpos[5:7] = [0.1, 0.0]
    data.qvel[3:7] = 0
    mujoco.mj_forward(model, data)
    prev_touch_pos = np.array([0.6, 0.0])

reset_game()
scene, camera, option = mujoco.MjvScene(model, 1000), mujoco.MjvCamera(), mujoco.MjvOption()
mujoco.mjv_defaultCamera(camera)
mujoco.mjv_defaultOption(option)

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    pyautogui.getWindowsWithTitle("MuJoCo")[0].maximize()
    while viewer.is_running():
        mouse_x, mouse_y = pyautogui.position()
        target_x = (mouse_x - 974) / 1000.0
        target_y = (2 * 519 - mouse_y) / 1000.0
        target_pos = np.clip([target_x, target_y], [-0.95, -0.45], [0.95, 0.45])
        velocity = np.clip((target_pos - prev_touch_pos) / env.dt, -0.3, 0.3)

        for _ in range(10):
            obs_raw = env._obs
            obs = strip_z(obs_raw)
            ppo_action, _ = ppo_trainer.act(obs)
            full_action = np.concatenate([ppo_action, velocity])
            ctrl = controller.apply_action(full_action)
            data.ctrl[:4] = ctrl[:4]

            next_obs_raw, _, _, _ = env.step(ppo_action)
            env._obs = next_obs_raw
            mujoco.mj_step(model, data)

        prev_touch_pos = target_pos

        if AirHockeyBase.is_absorbing_ui(strip_z(env._obs)):
            print("Goal scored! Resetting match.")
            reset_game()

        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewer.sync()
