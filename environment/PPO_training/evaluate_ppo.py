import time
import numpy as np
import mujoco
import mujoco.viewer
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from rule_based_ai_agent_v31 import AI_script_v31 as script
from environment.PPO_training.ppo_trainer import PPOTrainer

def strip_z(obs, env):
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

def reset_env_centered(env):
    obs = env.reset()
    env._data.qpos[:2] = 0
    env._data.qvel[:2] = 0
    mujoco.mj_forward(env._model, env._data)
    return obs

def main(render=True, episodes=50, delay=0.01):
    env = AirHockeyBase()
    controller = MalletControl(env_info=env.env_info, debug=False)
    model = env._model
    data = env._data

    trainer = PPOTrainer(obs_dim=8, action_dim=2)
    trainer.load("PPO_training_saved_models/saved_model")
    scripted_ai = script.startup()

    paddle_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
    puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

    results = []
    total_rewards = []

    def run_one_episode():
        obs = strip_z(reset_env_centered(env), env)
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < 500:
            puck_pos = float(data.xpos[puck_id][0]) * 1000 + 974, float(data.xpos[puck_id][1]) * 1000 + 519
            puck_pos_reverted = 2 * 974 - puck_pos[0], 2 * 519 - puck_pos[1]
            mallet2_pos = 2 * 974 - (float(data.xpos[paddle_id2][0]) * 1000 + 974), \
                          2 * 519 - (float(data.xpos[paddle_id2][1]) * 1000 + 519)

            action1, _ = trainer.act(obs)
            action2 = script.run(scripted_ai, puck_pos_reverted, mallet2_pos)
            action2 = np.array([-action2[0], -action2[1]])
            full_action = np.concatenate([action1, action2])

            control_action = controller.apply_action(full_action)
            data.ctrl[:4] = control_action[:4]

            next_obs_raw, reward, done, _ = env.step(action1)
            obs = strip_z(next_obs_raw, env)
            episode_reward += reward

            mujoco.mj_step(model, data)
            if render:
                viewer.sync()
                time.sleep(delay)

            steps += 1

        total_rewards.append(episode_reward)
        if reward > 0:
            results.append("Win")
        elif reward < 0:
            results.append("Loss")
        else:
            results.append("Draw")

    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for ep in range(episodes):
                print(f"Running evaluation episode {ep + 1}/{episodes}...")
                run_one_episode()
    else:
        for ep in range(episodes):
            run_one_episode()

    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {episodes}")
    print(f"Wins:   {results.count('Win')}")
    print(f"Losses: {results.count('Loss')}")
    print(f"Draws:  {results.count('Draw')}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    main(render=True, episodes=50, delay=0.01)