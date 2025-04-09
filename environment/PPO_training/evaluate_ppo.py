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

    #puck position, this overrides the randomized puck function in env_base, centers the puck infront of mallet
    env._data.qpos[:2] = 0
    env._data.qvel[:2] = 0
    mujoco.mj_forward(env._model, env._data)
    return obs


def main():
    env = AirHockeyBase()
    model = env._model
    data = env._data
    controller = MalletControl(env_info=env.env_info, debug=False)

    trainer = PPOTrainer(obs_dim=8, action_dim=2)
    trainer.load("PPO_training_saved_models/saved_model")

    scripted_ai = script.startup()

    paddle_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")
    puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

    obs = strip_z(reset_env_centered(env), env)
    done = False
    step = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
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

            next_obs_raw, _, done, _ = env.step(action1)
            obs = strip_z(next_obs_raw, env)

            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1
            if done or step > 500:
                step = 0
                obs = strip_z(reset_env_centered(env), env)
                time.sleep(1)


if __name__ == "__main__":
    main()
