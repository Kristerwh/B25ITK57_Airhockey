import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#For graphs and plotting
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime

import mujoco.viewer
import os
import numpy as np
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from rule_based_ai_agent_v31 import AI_script_v31 as script
from environment.neural_network.rlagent_leaky import RLAgent
from buffer import ReplayBuffer

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
controller = MalletControl(env_info=env_info, debug=True)

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

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f"runs/rl_airhockey_{run_id}")
reward_log = []
loss_log = []

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns)

def normalize(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1

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
    raw_obs = np.concatenate([puck_pos, puck_vel, mallet_pos, mallet_vel])
    normalized_obs = 2 * (raw_obs - env.obs_low) / (env.obs_high - env.obs_low) - 1
    return normalized_obs

try:
    with (mujoco.viewer.launch_passive(model, data) as viewer):
        # agent.load("saved/model_rl.keras")
        mujoco.mj_step(model, data)
        obs = strip_z(env.reset())
        step = 0
        episode = 0
        while viewer.is_running():
            base_pos = np.array([data.qpos[5], data.qpos[6]])
            puck_pos = float(data.xpos[puck_id][0] * 1000) + 974, float(data.xpos[puck_id][1] * 1000) + 519
            puck_pos_reverted = 2 * 974 - (float(data.xpos[puck_id][0] * 1000) + 974), 2 * 519 - (float(data.xpos[puck_id][1] * 1000) + 519)
            mallet_pos_script_ai = float(data.xpos[paddle_id][0] * 1000) + 974, float(data.xpos[paddle_id][1] * 1000) + 519
            mallet2_pos_script_ai = 2 * 974 - (float(data.xpos[paddle_id2][0] * 1000) + 974), 2 * 519 - (float(data.xpos[paddle_id2][1] * 1000) + 519)

            noise = np.random.normal(0, 1, size=2) if np.random.rand() < 0.005 else 0
            action1 = agent.predict(obs) + noise
            action2 = script.run(scripted_ai2, puck_pos_reverted, mallet2_pos_script_ai)
            action2 = np.array([-action2[0], -action2[1]])
            action = np.concatenate((action1, action2))

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

            step += 1

            if step % 500 == 0 or absorbing:
                if reward_buffer: # Only train if we have something
                    reward_buffer = [r if r is not None else 0.0 for r in reward_buffer]
                    returns = compute_returns(reward_buffer)
                    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
                    loss = agent.train(obs_buffer, action_buffer, returns)
                    print(f"Step {step} done. Reward sum: {np.sum(reward_buffer):.2f}, Loss: {loss:.4f}")

                    reward_log.append(np.sum(reward_buffer))
                    loss_log.append(loss)
                    writer.add_scalar("Reward", np.sum(reward_buffer), episode)
                    writer.add_scalar("Loss", loss, episode)

                    obs_buffer.clear()
                    action_buffer.clear()
                    reward_buffer.clear()

                if episode == 50:
                    agent.save('saved/model_rl.keras', include_optimizer=True)
                    print(f"Lagret modell etter {episode} episoder")
                    episode = 0

                if absorbing or step >= 500:
                    obs = strip_z(env.reset())
                    step = 0
                    episode += 1
                    print(f"Episode {episode} done.")
                obs = strip_z(next_obs)
            else:
                obs = strip_z(next_obs)

finally:
    writer.close()

    plot_dir = f"rl_training_plots/{run_id}"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(reward_log, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("RL Reward per Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{plot_dir}/reward_plot.png")
    plt.close()

    plt.figure()
    plt.plot(loss_log, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("RL Loss per Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{plot_dir}/loss_plot.png")
    plt.close()
