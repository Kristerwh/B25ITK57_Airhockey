import mujoco
import numpy as np
from position_control_wrapper import PositionControl


class MuJoCoPhysics:
    def __init__(self, xml_file):
        self.physics = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.physics)
        self.controller = PositionControl(p_gain=1.0, d_gain=0.1, i_gain=0.05)

    def step(self):
        self.controller.update_joint_positions()
        mujoco.mj_step(self.physics, self.data)

    def get_position(self, object_name):
        if object_name in ["paddle1", "paddle2"]:
            object_name += "_x"
        obj_id = mujoco.mj_name2id(self.physics, mujoco.mjtObj.mjOBJ_JOINT, object_name)
        if obj_id == -1:
            obj_id = mujoco.mj_name2id(self.physics, mujoco.mjtObj.mjOBJ_BODY, object_name)

        if obj_id == -1:
            raise ValueError({object_name})

        return np.array(self.data.qpos[obj_id:obj_id + 3])

    def set_position(self, object_name, new_position):
        if object_name in ["paddle1", "paddle2"]:
            object_name += "_x"
        obj_id = mujoco.mj_name2id(self.physics, mujoco.mjtObj.mjOBJ_JOINT, object_name)
        if obj_id == -1:
            obj_id = mujoco.mj_name2id(self.physics, mujoco.mjtObj.mjOBJ_BODY, object_name)

        if obj_id == -1:
            raise ValueError({object_name})

        self.data.qpos[obj_id:obj_id + len(new_position)] = np.array(new_position)

    def get_velocity(self, object_name):
        obj_id = mujoco.mj_name2id(self.physics, mujoco.mjtObj.mjOBJ_JOINT, object_name)
        if obj_id == -1:
            obj_id = mujoco.mj_name2id(self.physics, mujoco.mjtObj.mjOBJ_BODY, object_name)

        if obj_id == -1:
            raise ValueError({object_name})
        return self.data.qvel[obj_id]

    def set_velocity(self, object_name, new_velocity):
        obj_id = mujoco.mj_name2id(self.physics, mujoco.mjtObj.mjOBJ_JOINT, object_name)
        if obj_id == -1:
            obj_id = mujoco.mj_name2id(self.physics, mujoco.mjtObj.mjOBJ_BODY, object_name)

        if obj_id == -1:
            raise ValueError({object_name})
        self.data.qvel[obj_id] = new_velocity
