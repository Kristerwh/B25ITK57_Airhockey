import sys
import os
import time

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
agent.compile("adam", "mean_squared_error", "mean_absolute_error")
obs = env.reset()

with (mujoco.viewer.launch_passive(model, data) as viewer):
    step = 0
    while viewer.is_running():
        print("Initial obs:", obs)
        base_pos = np.array([data.qpos[5], data.qpos[6]])
        notpixel_pos = float(data.xpos[puck_id][0]), float(data.xpos[puck_id][1])
        print("notpixel_puck_pos:", notpixel_pos)
        puck_pos = float(data.xpos[puck_id][0] * 1000) + 974, float(data.xpos[puck_id][1] * 1000) + 519
        puck_pos_reverted = 2 * 974 - (float(data.xpos[puck_id][0] * 1000) + 974), 2 * 519 - (float(data.xpos[puck_id][1] * 1000) + 519)
        mallet_pos_script_ai = float(data.xpos[paddle_id][0] * 1000) + 974, float(data.xpos[paddle_id][1] * 1000) + 519
        mallet2_pos_script_ai = 2 * 974 - (float(data.xpos[paddle_id2][0] * 1000) + 974), 2 * 519 - (float(data.xpos[paddle_id2][1] * 1000) + 519)

        action1 = agent.predict(obs)
        action2 = script.run(scripted_ai2, puck_pos_reverted, mallet2_pos_script_ai)
        action2 = np.array([-action2[0], -action2[1]])
        action = np.concatenate((action1, action2))

        control_action = controller.apply_action(action)
        data.ctrl[:2] = control_action[:2]
        data.ctrl[2:4] = control_action[2:4]

        next_obs, reward, absorbing, _ = env.step(action1)

        if reward > 0:
            target_action = action1 + reward * 0.05
            agent.fit(np.array([obs]), np.array([target_action]), epochs=1)

        mujoco.mj_step(model, data)

        if step % 10 == 0:
            mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            viewer.sync()

        obs = next_obs
        step += 1

        if absorbing:
            obs = env.reset()
