import mujoco
import mujoco.viewer
import os
import numpy as np
from environment.env_settings.environments.position_control_wrapper import PositionControl
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase

env = AirHockeyBase()

env_info = env.env_info

n_agents = env_info.get("n_agents", 1)

env_info["actuator_joint_ids"] = env_info.get("actuator_joint_ids", [6, 7, 8, 9, 10, 11, 12])
env_info["_timestep"] = env_info.get("dt", 0.02)

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "env_settings", "environments", "data", "table.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

scene = mujoco.MjvScene(model, maxgeom=1000)
perturb = mujoco.MjvPerturb()
mujoco.mjv_defaultPerturb(perturb)

camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)

option = mujoco.MjvOption()
option.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

p_gain, d_gain, i_gain = 10.0, 1.0, 0.1
controller = PositionControl(p_gain, d_gain, i_gain, env_info=env_info)

paddle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_left")
puck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        #paddle and puck positions, we won't need these when we add AI hands
        paddle_pos = data.qpos[paddle_id * 7: (paddle_id + 1) * 7]
        puck_pos = data.qpos[puck_id * 7: (puck_id + 1) * 7]

        obs = data.qpos[:]
        action = np.zeros((20, 3, controller.n_robot_joints))
        valid_joint_ids = [idx for idx in controller.actuator_joint_ids if idx < len(data.qpos)]
        valid_robot_joint_ids = [idx for idx in controller.actuator_joint_ids if idx < len(data.qpos) and idx < len(data.qvel)]
        cur_pos = np.zeros(len(controller.actuator_joint_ids))
        for i, idx in enumerate(valid_robot_joint_ids):
            if idx < len(data.qpos):
                cur_pos[i] = data.qpos[idx]
        cur_vel = np.zeros(len(controller.actuator_joint_ids))
        for i, idx in enumerate(valid_robot_joint_ids):
            if idx < len(data.qvel):
                cur_vel[i] = data.qvel[idx]

        #Temporary placeholders for desired trajectory we can replace with real trajectory computation later
        desired_pos = cur_pos
        desired_vel = cur_vel
        desired_acc = np.zeros_like(cur_pos)

        control_action = controller._controller(desired_pos, desired_vel, desired_acc, cur_pos, cur_vel)
        if control_action.shape != data.ctrl.shape:
            control_action = np.pad(control_action, (0, max(0, data.ctrl.shape[0] - control_action.shape[0])))

        if control_action.shape[0] > data.ctrl.shape[0]:
            control_action = control_action[:data.ctrl.shape[0]]
        elif control_action.shape[0] < data.ctrl.shape[0]:
            control_action = np.pad(control_action,(0, data.ctrl.shape[0] - control_action.shape[0]))

        data.ctrl[:] = 0

        mujoco.mj_step(model, data)
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewer.sync()
