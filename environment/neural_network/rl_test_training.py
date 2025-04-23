import sys
import os
import numpy as np
import mujoco
import mujoco.viewer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.neural_network.rlagent import RLAgent
from rule_based_ai_agent_v31 import AI_script_v31 as script

# Initialize environment and models
env = AirHockeyBase()
model = env._model
data = env._data

paddle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_left")
paddle_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

controller = MalletControl(env_info=env.env_info, debug=False)
scripted_ai = script.startup()
scripted_ai2 = script.startup()

input_shape = 8  # puck(x,y,vel_x,vel_y) + mallet(x,y,vel_x,vel_y)
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

def render_episode(viewer):
    obs = strip_z(env.reset())
    step = 0
    done = False
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None,
                           mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL,
                           mujoco.MjvScene(model, 1000))
    viewer.sync()
    import time
    time.sleep(3)

    while viewer.is_running() and not done and step < 2000:
        puck_pos = float(data.xpos[puck_id][0]) * 1000 + 974, float(data.xpos[puck_id][1]) * 1000 + 519
        puck_pos_reverted = 2 * 974 - puck_pos[0], 2 * 519 - puck_pos[1]
        mallet2_pos_script_ai = 2 * 974 - (float(data.xpos[paddle_id2][0]) * 1000 + 974), \
                                2 * 519 - (float(data.xpos[paddle_id2][1]) * 1000 + 519)

        action1 = agent.predict(obs)
        action2 = script.run(scripted_ai2, puck_pos_reverted, mallet2_pos_script_ai)
        action2 = np.array([-action2[0], -action2[1]])
        action = np.concatenate((action1, action2))

        control_action = controller.apply_action(action)
        data.ctrl[:2] = control_action[:2]
        data.ctrl[2:4] = control_action[2:4]

        next_obs_raw, _, absorbing, _ = env.step(action1)
        mujoco.mj_step(model, data)
        obs = strip_z(next_obs_raw)

        if step % 10 == 0:
            mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, mujoco.MjvScene(model, 1000))
            viewer.sync()

        step += 1
        done = absorbing

# Training loop
viewer = mujoco.viewer.launch_passive(model, data)
agent.load("saved/model_rl.keras")
num_episodes = 1000
for episode in range(num_episodes + 1):
    obs_raw = env.reset()
    mujoco.mj_step(model, data)
    obs = strip_z(obs_raw)

    obs_buffer.clear()
    action_buffer.clear()
    reward_buffer.clear()

    step = 0
    done = False

    while not done and step < 2500:
        puck_pos = float(data.xpos[puck_id][0]) * 1000 + 974, float(data.xpos[puck_id][1]) * 1000 + 519
        puck_pos_reverted = 2 * 974 - puck_pos[0], 2 * 519 - puck_pos[1]
        mallet2_pos_script_ai = 2 * 974 - (float(data.xpos[paddle_id2][0]) * 1000 + 974), \
                                2 * 519 - (float(data.xpos[paddle_id2][1]) * 1000 + 519)

        noise = np.random.normal(0, 1, size=2) if np.random.rand() < 0.1 else 0
        action1 = agent.predict(obs) + noise

        action2 = script.run(scripted_ai2, puck_pos_reverted, mallet2_pos_script_ai)
        action2 = np.array([-action2[0], -action2[1]])
        action = np.concatenate((action1, action2))

        control_action = controller.apply_action(action)
        data.ctrl[:2] = control_action[:2]
        data.ctrl[2:4] = control_action[2:4]

        next_obs_raw, reward, absorbing, _ = env.step(action1)
        mujoco.mj_step(model, data)

        obs_buffer.append(obs)
        action_buffer.append(action1)
        reward_buffer.append(reward)

        obs = strip_z(next_obs_raw)
        step += 1
        done = absorbing

    reward_buffer = [r if r is not None else 0.0 for r in reward_buffer]
    returns = compute_returns(reward_buffer)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    loss = agent.train(obs_buffer, action_buffer, returns)

    print(f"Episode {episode} done. Total reward: {np.sum(reward_buffer):.2f}, Loss: {loss:.4f}")

    if episode % 25 == 0:
        agent.save('saved/model_rl.keras', include_optimizer=True)
        print(f"Model saved after episode {episode}")

    if episode % 50 == 0:
        print("Rendering episode for evaluation...")
        render_episode(viewer)