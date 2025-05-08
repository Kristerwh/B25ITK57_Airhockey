import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from environment.env_settings.environments.position_controller_mallet_wrapper import MalletControl
from environment.PPO_training.ppo_trainer import PPOTrainer
from environment.PPO_training.ppo_rewards2 import phase1_reward, phase2_reward, phase3_reward
#phase4_reward, phase5_reward

from rule_based_ai_agent_v31 import AI_script_v31 as script
from environment.env_settings.environments.iiwas.env_base import is_colliding_ppo
from environment.env_settings.environments.position_controller_mallet_wrapper import apply_action_ppo
import mujoco
import mujoco.viewer
from torch.utils.tensorboard import SummaryWriter

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

def reset_env(env, randomize_puck=True):
    obs = env.reset()

    if randomize_puck:
        puck_x = np.random.uniform(-0.4, -0.1)
        puck_y = np.random.uniform(-0.2, 0.4)
    else:
        puck_x = -0.3
        puck_y = 0.0

    mallet_x = -0.10
    mallet_y = 0.0

    # Set positions
    env._data.qpos[env._model.jnt("puck_x").qposadr] = puck_x
    env._data.qpos[env._model.jnt("puck_y").qposadr] = puck_y
    env._data.qpos[env._model.jnt("paddle_left_x").qposadr] = mallet_x
    env._data.qpos[env._model.jnt("paddle_left_y").qposadr] = mallet_y

    # Zero velocities
    env._data.qvel[env._model.jnt("puck_x").dofadr] = 0
    env._data.qvel[env._model.jnt("puck_y").dofadr] = 0
    env._data.qvel[env._model.jnt("paddle_left_x").dofadr] = 0
    env._data.qvel[env._model.jnt("paddle_left_y").dofadr] = 0

    mujoco.mj_forward(env._model, env._data)
    return env._create_observation(env.obs_helper._build_obs(env._data))

def main(render=True):
    env = AirHockeyBase()
    controller = MalletControl(env_info=env.env_info, debug=False)
    model = env._model
    data = env._data

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_dir = f"ppo_training_plots/{run_id}"
    os.makedirs(plot_dir, exist_ok=True)
    writer = SummaryWriter(f"runs/ppo_airhockey_{run_id}")

    scripted_ai = script.startup()
    paddle_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
    puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

    obs_dim = 8
    act_dim = 2
    trainer = PPOTrainer(obs_dim, act_dim)
    total_episodes = 1000
    rollout_len = 500

    reward_log = []
    policy_losses, value_losses, entropies = [], [], []

    def run_episode(ep):
        env.prev_action = np.zeros(2)

        use_random_puck = ep < 500
        obs = strip_z(reset_env(env, randomize_puck=use_random_puck), env)

        if ep >= 500:
            env._data.qpos[0:2] = -0.3, 0.0
            env._data.qvel[0:2] = 0, 0
            mujoco.mj_forward(env._model, env._data)

        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        episode_reward = 0
        use_scripted = (ep >= 50)

        for step in range(rollout_len):
            action1, log_prob = trainer.act(obs)
            action1 = np.clip(action1, -100, 100)
            value = trainer.evaluate(obs)

            puck_pos = float(data.xpos[puck_id][0]) * 1000 + 974, float(data.xpos[puck_id][1]) * 1000 + 519
            puck_pos_reverted = 2 * 974 - puck_pos[0], 2 * 519 - puck_pos[1]
            mallet2_pos = 2 * 974 - (float(data.xpos[paddle_id2][0]) * 1000 + 974), \
                          2 * 519 - (float(data.xpos[paddle_id2][1]) * 1000 + 519)

            script_noise = 0.05 if ep >= 1500 else 0.0
            if use_scripted:
                action2 = script.run(scripted_ai, puck_pos_reverted, mallet2_pos)
                action2 = np.array(action2[:2], dtype=np.float32)
                action2 += np.random.normal(0, script_noise, size=2)
            else:
                action2 = np.zeros(2, dtype=np.float32)

            action2 = -action2

            full_action = np.concatenate([action1, action2])
            control = apply_action_ppo(full_action)

            data.ctrl[:] = control

            next_obs_raw, _, done, _ = env.step(action1)
            next_obs = strip_z(next_obs_raw, env)

            #phase blending (episodes: 0–500 → phase1 to phase2, 500–1000 → phase2 to phase3)
            if ep < 500:
                alpha = ep / 500
                r1 = phase1_reward(obs, action1, next_obs, done, env, ep=ep)
                r2 = phase2_reward(obs, action1, next_obs, done, env, ep=ep)
                reward = (1 - alpha) * r1 + alpha * r2
            elif ep < 1000:
                alpha = (ep - 500) / 500
                r2 = phase2_reward(obs, action1, next_obs, done, env, ep=ep)
                r3 = phase3_reward(obs, action1, next_obs, done, env, ep=ep)
                reward = (1 - alpha) * r2 + alpha * r3
            else:
                reward = phase3_reward(obs, action1, next_obs, done, env, ep=ep)

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

            if is_colliding_ppo(next_obs[0:2], next_obs[4:6]):  # <- REPLACED
                writer.add_scalar("Hit", 1, ep)
            if next_obs[0] > 0 and done:
                writer.add_scalar("Goal", 1, ep)
            if next_obs[0] < 0 and done:
                writer.add_scalar("Conceded", 1, ep)

            if done:
                break

        reward_log.append(episode_reward)
        advantages, returns = trainer.compute_gae(rew_buf, val_buf, done_buf)
        pol_loss, val_loss, entropy_val = trainer.ppo_update(obs_buf, act_buf, logp_buf, returns, advantages, current_episode=ep)
        policy_losses.append(pol_loss)
        value_losses.append(val_loss)
        entropies.append(entropy_val)

        writer.add_scalar("Reward", episode_reward, ep)
        writer.add_scalar("Policy Loss", pol_loss, ep)
        writer.add_scalar("Value Loss", val_loss, ep)
        writer.add_scalar("Entropy", entropy_val, ep)

    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for episode in trange(total_episodes):
                run_episode(episode)
                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(reward_log[-50:])
                    print(f"Episode {episode + 1}: avg reward (last 50 eps) = {avg_reward:.2f}")
                    trainer.save("PPO_training_saved_models/saved_model")
    else:
        for episode in trange(total_episodes):
            run_episode(episode)
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(reward_log[-50:])
                print(f"Episode {episode + 1}: avg reward (last 50 eps) = {avg_reward:.2f}")
                trainer.save("PPO_training_saved_models/saved_model")

    plt.figure()
    plt.plot(reward_log, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Reward Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/ppo_training_reward_curve.png")
    plt.close()

    plt.figure()
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.plot(entropies, label='Entropy')
    plt.xlabel('Episode')
    plt.title('PPO Losses & Entropy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/ppo_training_diagnostics.png")
    plt.close()

    writer.close()

if __name__ == "__main__":
    main(render=True)
