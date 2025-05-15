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

    #puck and mallet positions
    env._data.qpos[env._model.jnt("puck_x").qposadr] = -0.3
    env._data.qpos[env._model.jnt("puck_y").qposadr] = 0.0
    env._data.qpos[env._model.jnt("paddle_left_x").qposadr] = -0.1
    env._data.qpos[env._model.jnt("paddle_left_y").qposadr] = 0.0

    env._data.qvel[env._model.jnt("puck_x").dofadr] = 0
    env._data.qvel[env._model.jnt("puck_y").dofadr] = 0
    env._data.qvel[env._model.jnt("paddle_left_x").dofadr] = 0
    env._data.qvel[env._model.jnt("paddle_left_y").dofadr] = 0

    mujoco.mj_forward(env._model, env._data)
    return env._create_observation(env.obs_helper._build_obs(env._data))

def main(render=True, episodes=50):
    env = AirHockeyBase()
    controller = MalletControl(env_info=env.env_info, debug=False)
    model = env._model
    data = env._data

    trainer = PPOTrainer(obs_dim=8, action_dim=2)
    #trainer.load("PPO_training_saved_models/phase3/saved_model")
    trainer.load("PPO_training_saved_models/saved_model")
    scripted_ai = script.startup()

    paddle_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
    puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

    results = []
    total_rewards = []
    total_hits = 0
    total_goals_scored = 0
    total_goals_conceded = 0

    def run_one_episode():
        nonlocal total_hits, total_goals_scored, total_goals_conceded

        obs = strip_z(reset_env_centered(env), env)
        episode_reward = 0
        hit_count = 0
        goal_scored = 0
        goal_conceded = 0
        steps_since_last_hit = 0
        done = False
        max_steps = 5000

        for step in range(max_steps):
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

            next_obs_raw, reward, _, _ = env.step(action1)
            obs = strip_z(next_obs_raw, env)

            if hasattr(env, "is_colliding") and env.is_colliding(next_obs_raw, 'puck', 'paddle_left'):
                hit_count += 1
                steps_since_last_hit = 0
            else:
                steps_since_last_hit += 1

            puck_x = float(data.xpos[puck_id][0])
            if puck_x > 0.95:
                goal_scored += 1
                done = True
            elif puck_x < -0.95:
                goal_conceded += 1
                done = True
            elif steps_since_last_hit > 10000:
                done = True

            episode_reward += reward
            mujoco.mj_step(model, data)
            if render:
                viewer.sync()

            if done:
                break

        total_rewards.append(episode_reward)
        total_hits += hit_count
        total_goals_scored += goal_scored
        total_goals_conceded += goal_conceded

        if goal_scored > goal_conceded:
            results.append("Win")
        elif goal_scored < goal_conceded:
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
    print(f"Average Hits per Episode: {total_hits / episodes:.2f}")
    print(f"Goals Scored per Episode: {total_goals_scored / episodes:.2f}")
    print(f"Goals Conceded per Episode: {total_goals_conceded / episodes:.2f}")

if __name__ == "__main__":
    main(render=True, episodes=50)
