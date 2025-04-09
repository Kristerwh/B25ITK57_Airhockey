import os
import numpy as np
from tqdm import trange
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.PPO_training.ppo_trainer import PPOTrainer
from rule_based_ai_agent_v31 import AI_script_v31 as script
import mujoco
import mujoco.viewer


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
    env._data.qpos[:2] = 0  # Reset puck x, y to center (adjust if needed)
    env._data.qvel[:2] = 0  # Stop puck velocity
    mujoco.mj_forward(env._model, env._data)
    return obs


def main(render=True):
    env = AirHockeyBase()
    controller = MalletControl(env_info=env.env_info, debug=False)
    model = env._model
    data = env._data

    scripted_ai = script.startup()
    paddle_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
    puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

    obs_dim = 8
    act_dim = 2

    trainer = PPOTrainer(obs_dim, act_dim)
    total_episodes = 1000
    rollout_len = 200

    reward_log = []

    def run_episode():
        obs = strip_z(reset_env_centered(env), env)
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        episode_reward = 0

        for step in range(rollout_len):
            # PPO agent action
            action1, log_prob = trainer.act(obs)
            value = trainer.evaluate(obs)

            # Scripted AI action (right mallet)
            puck_pos = float(data.xpos[puck_id][0]) * 1000 + 974, float(data.xpos[puck_id][1]) * 1000 + 519
            puck_pos_reverted = 2 * 974 - puck_pos[0], 2 * 519 - puck_pos[1]
            mallet2_pos = 2 * 974 - (float(data.xpos[paddle_id2][0]) * 1000 + 974), \
                          2 * 519 - (float(data.xpos[paddle_id2][1]) * 1000 + 519)
            action2 = script.run(scripted_ai, puck_pos_reverted, mallet2_pos)
            action2 = np.array([-action2[0], -action2[1]])

            full_action = np.concatenate([action1, action2])
            control = controller.apply_action(full_action)
            data.ctrl[:] = control

            next_obs_raw, reward, done, _ = env.step(action1)
            next_obs = strip_z(next_obs_raw, env)

            episode_reward += reward

            obs_buf.append(obs)
            act_buf.append(action1)
            logp_buf.append(log_prob)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(done)

            obs = next_obs

            if render:
                mujoco.mj_step(model, data)
                viewer.sync()

            if done:
                break

        reward_log.append(episode_reward)
        advantages, returns = trainer.compute_gae(rew_buf, val_buf, done_buf)
        trainer.ppo_update(obs_buf, act_buf, logp_buf, returns, advantages)

    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for episode in trange(total_episodes):
                run_episode()
                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(reward_log[-50:])
                    print(f"Episode {episode + 1}: avg reward (last 50 eps) = {avg_reward:.2f}")
                    trainer.save("PPO_training_saved_models/saved_model")
                    print(f"Saved model at episode {episode + 1}")
    else:
        for episode in trange(total_episodes):
            run_episode()
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(reward_log[-50:])
                print(f"Episode {episode + 1}: avg reward (last 50 eps) = {avg_reward:.2f}")
                trainer.save("PPO_training_saved_models/saved_model")
                print(f"Saved model at episode {episode + 1}")


if __name__ == "__main__":
    main(render=True)
