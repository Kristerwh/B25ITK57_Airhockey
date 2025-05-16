import sys
import os

from environment.PPO_training.ppo_agent import PPOAgent
from environment.PPO_training.ppo_trainer import PPOTrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#For graphs and plotting
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time

import mujoco.viewer
import os
import numpy as np
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from rule_based_ai_agent_v31 import AI_script_v31 as script
from environment.neural_network.rlagent_leaky import RLAgent
import tensorflow as tf
# from buffer import ReplayBuffer

env = AirHockeyBase(n_agents=1)

env_info = env.env_info

n_agents = env_info.get("n_agents", 1)

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


input_shape = len(env.get_all_observation_keys()) + 2
action_dim = 2

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

def normalize(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1

from collections import deque
def strip_z(obs, normalized=True):
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
    if normalized:
        return normalized_obs
    else:
        return raw_obs

from collections import deque
sequence_length = 10
obs_sequence = deque(maxlen=sequence_length)

agent = RLAgent(sequence_length=10, input_shape=8)
agent.load("neural_network/saved/model_rl.keras")
ppoagent = PPOTrainer(obs_dim=8, action_dim=2)
ppoagent.load("PPO_training/PPO_training_saved_models/saved_model")
scripted_ai = script.startup()
scripted_ai2 = script.startup()
chosen_agent = scripted_ai

# Set up TensorBoard writer
run_id = time.strftime("%Y%m%d-%H%M%S")
log_dir = f"tensorboard_logs/{run_id}"
writer = tf.summary.create_file_writer(log_dir)

# Initialize result logs
goal_log = []
opponent_goal_log = []
time_to_first_goal_log = []
goal_times_log = []
goal_type_log = []

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:

        step = 0
        episode = 1
        goals = 0

        obs_sequence = deque(maxlen=sequence_length)
        if episode % 2 == 0:
            obs_raw = strip_z(env.reset("opponent_y"), normalized=False)
            obs_normalized = strip_z(env.reset("opponent_y"))
        else:
            obs_raw = strip_z(env.reset("y"), normalized=False)
            obs_normalized = strip_z(env.reset("y"))
        for _ in range(sequence_length):
            obs_sequence.append(obs_normalized)
        obs = np.array(obs_sequence)

        # Set initial timers
        episode_start_time = time.time()
        goals_in_episode = 0
        opponent_goals_in_episode = 0
        first_goal_time = None

        while viewer.is_running():
            goals_in_episode = 0
            opponent_goals_in_episode = 0
            first_goal_time = None
            if goals >= 10:
                print("10 episodes completed. Exiting...")
                break
            base_pos = np.array([data.qpos[5], data.qpos[6]])
            puck_pos = float(data.xpos[puck_id][0] * 1000) + 974, float(data.xpos[puck_id][1] * 1000) + 519
            puck_pos_reverted = 2 * 974 - (float(data.xpos[puck_id][0] * 1000) + 974), 2 * 519 - (float(data.xpos[puck_id][1] * 1000) + 519)
            mallet_pos_script_ai = float(data.xpos[paddle_id][0] * 1000) + 974, float(data.xpos[paddle_id][1] * 1000) + 519
            mallet2_pos_script_ai = 2 * 974 - (float(data.xpos[paddle_id2][0] * 1000) + 974), 2 * 519 - (float(data.xpos[paddle_id2][1] * 1000) + 519)

            if chosen_agent == scripted_ai:
                action = script.run(scripted_ai, puck_pos, mallet_pos_script_ai)
                action = np.asarray(action).reshape(-1)[:2]
                # data.ctrl[:2] = action[:2]
                # mujoco.mj_step(model, data)
                env.step(action)
            elif chosen_agent == agent:
                action = agent.predict(obs.reshape(1, sequence_length, 8))
                action = np.asarray(action).reshape(-1)
                action = np.clip(action, -0.1, 0.1)
                next_obs, reward, absorbing, _ = env.step(action)
            elif chosen_agent == ppoagent:
                action, _ = ppoagent.act(obs_raw)
                next_obs, reward, absorbing, _ = env.step(action)

            opponent_action = script.run(scripted_ai2, puck_pos_reverted, mallet2_pos_script_ai)
            opponent_action = -opponent_action[0], -opponent_action[1]
            opponent_action = np.asarray(opponent_action).reshape(-1)[:2]
            # opponent_action = np.clip(opponent_action, -0.15, 0.15)
            data.ctrl[2:4] = opponent_action[:2]

            if step % 1 == 0:
                mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                viewer.sync()

            step += 1

            current_time = time.time()
            if puck_pos[0] < 0 or puck_pos[0] > 1948 or step >= 10000:
                if puck_pos[0] > 1948:  # Agent scores
                    print("Agent scores")
                    goals_in_episode += 1
                    if first_goal_time is None:
                        first_goal_time = current_time - episode_start_time
                    # Log time and type
                    goal_times_log.append(current_time - episode_start_time)
                    goal_type_log.append("agent")
                    goals += 1


                elif puck_pos[0] < 0:  # Opponent scores

                    print("Opponent scores")

                    opponent_goals_in_episode += 1

                    if first_goal_time is None:
                        first_goal_time = current_time - episode_start_time

                    # Log time and type

                    goal_times_log.append(current_time - episode_start_time)

                    goal_type_log.append("opponent")
                    goals += 1

                elif step >= 10000:
                    print("reset")

                goal_log.append(goals_in_episode)
                opponent_goal_log.append(opponent_goals_in_episode)  # <<< ADDED
                if first_goal_time is not None:
                    time_to_first_goal_log.append(first_goal_time)
                else:
                    time_to_first_goal_log.append(float('inf'))  # If never scored

                # Log til TensorBoard
                with writer.as_default():
                    tf.summary.scalar("Goals per Episode", goals_in_episode, step=episode)
                    tf.summary.scalar("Opponent Goals per Episode", opponent_goals_in_episode, step=episode)  # <<< ADDED
                    tf.summary.scalar("Time to First Goal", first_goal_time if first_goal_time else float('inf'), step=episode)

                # Reset for neste episode
                print(f"Episode {episode} done.")
                episode += 1

                obs_sequence = deque(maxlen=sequence_length)
                if episode % 2 == 0:
                    obs_raw = strip_z(env.reset("opponent_y"), normalized=False)
                    obs_normalized = strip_z(env.reset("opponent_y"))
                else:
                    obs_raw = strip_z(env.reset("y"), normalized=False)
                    obs_normalized = strip_z(env.reset("y"))
                for _ in range(sequence_length):
                    obs_sequence.append(obs_normalized)
                obs = np.array(obs_sequence)

                step = 0
                episode_start_time = time.time()
                goals_in_episode = 0
                opponent_goals_in_episode = 0  # <<< Reset opponent goals
                first_goal_time = None

            else:
                if chosen_agent != scripted_ai:
                    obs_raw = strip_z(next_obs, normalized=False)
                    obs_normalized = strip_z(next_obs)
                    obs_sequence.append(obs_normalized)
                    obs = np.array(obs_sequence)

finally:
    writer.close()

    plot_dir = f"goal_plots/{run_id}"
    os.makedirs(plot_dir, exist_ok=True)

    # --- Save original goal plot ---
    plt.figure()
    plt.plot(goal_log, label="Agent Goals per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Goals")
    plt.title("Agent Goals per Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{plot_dir}/goal_plot.png")
    plt.close()

    # --- Save opponent goals plot ---
    plt.figure()
    plt.plot(opponent_goal_log, label="Opponent Goals per Episode", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Goals Conceded")
    plt.title("Opponent Goals per Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{plot_dir}/opponent_goal_plot.png")
    plt.close()

    # --- Save combined plot ---
    plt.figure()
    plt.plot(goal_log, label="Agent Goals", linestyle='-')
    plt.plot(opponent_goal_log, label="Opponent Goals", linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Goals")
    plt.title("Agent vs Opponent Goals per Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{plot_dir}/combined_goal_plot.png")
    plt.close()

    # --- Save time to first goal plot ---
    plt.figure()
    plt.plot(time_to_first_goal_log, label="Time to First Goal (s)")
    plt.xlabel("Episode")
    plt.ylabel("Seconds")
    plt.title("Time to First Goal per Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{plot_dir}/time_to_first_goal_plot.png")
    plt.close()

    # --- Save timeline scatter plot ---
    plt.figure()
    agent_label_added = False
    opponent_label_added = False
    for t, who in zip(goal_times_log, goal_type_log):
        if who == "agent":
            plt.scatter(t, 1, color='blue', label='Agent Goal' if not agent_label_added else "")
            agent_label_added = True
        else:
            plt.scatter(t, -1, color='red', label='Opponent Goal' if not opponent_label_added else "")
            opponent_label_added = True
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Time since Episode Start (s)")
    plt.ylabel("Goal Type (Agent=1, Opponent=-1)")
    plt.title("Timeline of Goals during Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/goal_timeline_plot.png")
    plt.close()
