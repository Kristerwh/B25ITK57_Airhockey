import sys
import os
import time
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mujoco
import mujoco.viewer
import os
import numpy as np
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from rule_based_ai_agent_v31 import AI_script_v31 as script
from environment.rlagent import RLAgent

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
input_shape = len(env.get_all_observation_keys()) + 2
action_dim = 2

agent = RLAgent(input_shape)
obs_buffer = []
action_buffer = []
reward_buffer = []


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns)


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


with (mujoco.viewer.launch_passive(model, data) as viewer):
    # agent.load("saved/model_rl.keras")
    mujoco.mj_step(model, data)
    obs = strip_z(env.reset())
    step = 0
    episode = 0
    while viewer.is_running():
        base_pos = np.array([data.qpos[5], data.qpos[6]])
        puck_pos = float(data.xpos[puck_id][0] * 1000) + 974, float(data.xpos[puck_id][1] * 1000) + 519
        puck_pos_reverted = 2 * 974 - (float(data.xpos[puck_id][0] * 1000) + 974), 2 * 519 - (
                    float(data.xpos[puck_id][1] * 1000) + 519)
        mallet_pos_script_ai = float(data.xpos[paddle_id][0] * 1000) + 974, float(data.xpos[paddle_id][1] * 1000) + 519
        mallet2_pos_script_ai = 2 * 974 - (float(data.xpos[paddle_id2][0] * 1000) + 974), 2 * 519 - (
                    float(data.xpos[paddle_id2][1] * 1000) + 519)

        noise = np.random.normal(0, 1, size=2) if np.random.rand() < 0.05 else 0
        action1 = agent.predict(obs) # + noise
        action2 = script.run(scripted_ai2, puck_pos_reverted, mallet2_pos_script_ai)
        action2 = np.array([-action2[0], -action2[1]])
        action = np.concatenate((action1, action2))

        # Sending AI's actions to the controller
        control_action = controller.apply_action(action)
        data.ctrl[:2] = control_action[:2]
        data.ctrl[2:4] = control_action[2:4]

        next_obs, reward, absorbing, _ = env.step(action1)
        mujoco.mj_step(model, data)
        obs_buffer.append(obs)
        action_buffer.append(action1)
        reward_buffer.append(reward)

        if step % 10 == 0:
            mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            viewer.sync()

        # if step % 1000 == 0:
        #     print(obs)
        step += 1

        if step % 250 == 0 or absorbing:
            if reward_buffer:  # Only train if we have something
                reward_buffer = [r if r is not None else 0.0 for r in reward_buffer]
                returns = compute_returns(reward_buffer)
                returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
                loss = agent.train(obs_buffer, action_buffer, returns)
                print(f"Step {step} done. Reward sum: {np.sum(reward_buffer):.2f}, Loss: {loss:.4f}")

                obs_buffer.clear()
                action_buffer.clear()
                reward_buffer.clear()

            if episode == 50:
                agent.save('saved/model_rl.keras', include_optimizer=True)
                print(f"✅ Lagret modell etter {episode} episoder")
                episode = 0

            # Reset buffers og miljø
            if absorbing or step >= 10000:
                obs = strip_z(env.reset())
                step = 0
                episode += 1
                print(f"Episode {episode} done. Total reward: {np.sum(reward_buffer):.2f}, Loss: {loss:.4f}")
            obs = strip_z(next_obs)
        else:
            obs = strip_z(next_obs)