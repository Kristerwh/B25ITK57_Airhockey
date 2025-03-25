import mujoco
import mujoco.viewer
import numpy as np
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from stable_baselines3 import PPO
from airhocky_manual_ai_v31 import main as manual_script
import os

env = AirHockeyBase()
model = PPO.load("rl_training/ppo_airhockey")
controller = MalletControl(env_info=env.env_info, debug=False)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "env_settings", "environments", "data", "table.xml")
mujoco_model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(mujoco_model)
paddle_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "paddle_left")
paddle_id2 = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
puck_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "puck")

manual_ai = manual_script.startup()

total_reward = 0.0
goals_scored = 0
own_goals = 0

#render loop
with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
    obs = env.reset()
    while viewer.is_running():
        puck_pos = float(data.xpos[puck_id][0]), float(data.xpos[puck_id][1])
        manual_pos = float(data.xpos[paddle_id2][0]), float(data.xpos[paddle_id2][1])
        mallet_action_manual = manual_script.run(manual_ai, puck_pos, manual_pos)
        mallet_action_manual = np.array([-mallet_action_manual[0], -mallet_action_manual[1]])

        mallet_action_rl, _ = model.predict(obs)

        action = np.concatenate([mallet_action_rl, mallet_action_manual])
        control_action = controller.apply_action(action)

        data.ctrl[:4] = control_action[:4]
        mujoco.mj_step(mujoco_model, data)

        obs, reward, done, _ = env.step(mallet_action_rl)
        total_reward += reward

        if reward > 0.9:
            goals_scored += 1
            print("+1 RL Agent")
        elif reward < -0.9:
            own_goals += 1
            print("-1 RL Agent")

        print(f"Current Reward: {reward:.3f} | Total: {total_reward:.2f} | Goals: {goals_scored} | Own Goals: {own_goals}")

        if done:
            print("Episode done, resetting environment.\n")
            obs = env.reset()
            total_reward = 0

        viewer.sync()
