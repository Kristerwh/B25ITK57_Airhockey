import mujoco
import numpy as np
import pygame

from constants import FRICTION, ELASTICITY, puck_radius, paddle_radius, puck_start_location, puck_speed


class Puck:
    def __init__(self, physics, body_name):
        self.physics = physics
        self.body_name = body_name

    def get_position(self):
        return self.physics.get_position(self.body_name)

    #temporary function to reset the puck, we'll expand on this later with goals and scores and more stuff
    def puck_reset(self):
        puck_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        self.physics.data.qpos[self.physics.model.jnt_qposadr[puck_id]:self.physics.model.jnt_qposadr[puck_id] + 3] = puck_start_location
        self.physics.data.qvel[self.physics.model.jnt_dofadr[puck_id]:self.physics.model.jnt_dofadr[puck_id] + 3] = [0, 0, 0]

    #temporary scoring function, just prints out which wall puck touched
    def point_scored(self):
        puck_pos = self.get_position()
        if -0.65 < puck_pos[0] <= -0.63 and abs(puck_pos[1]) <= 0.1:
            return "puck touched left wall"
        elif 0.63 <= puck_pos[0] < 0.65 and abs(puck_pos[1]) <= 0.1:
            return "puck touched right wal"
        return None

    def apply_force(self, force):
        self.physics.apply_force(self.body_name, force)

    def apply_friction(self):
        puck_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        dof_address = self.physics.model.body_dofadr[puck_id]

        self.physics.data.qvel[dof_address] *= FRICTION
        self.physics.data.qvel[dof_address + 1] *= FRICTION

    def check_collision(self, other_body_name):
        puck_pos = np.array(self.get_position())
        other_pos = np.array(self.physics.get_position(other_body_name))
        distance = np.linalg.norm(puck_pos - other_pos)
        return distance <= (puck_radius + paddle_radius)

    def handle_wall_collisions(self):
        puck_pos = self.get_position()
        puck_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        dof_address = self.physics.model.body_dofadr[puck_id]

        min_x, max_x = -0.6, 0.6
        min_y, max_y = -0.4, 0.4

        nudge = 0.002

        if puck_pos[0] - puck_radius / 1000 <= min_x:
            self.physics.data.qvel[dof_address] = abs(self.physics.data.qvel[dof_address]) * ELASTICITY
            self.physics.data.xpos[puck_id][0] = min_x + puck_radius / 1000 + nudge
        elif puck_pos[0] + puck_radius / 1000 >= max_x:
            self.physics.data.qvel[dof_address] = -abs(self.physics.data.qvel[dof_address]) * ELASTICITY
            self.physics.data.xpos[puck_id][0] = max_x - puck_radius / 1000 - nudge

        if puck_pos[1] - puck_radius / 1000 <= min_y:
            self.physics.data.qvel[dof_address + 1] = abs(self.physics.data.qvel[dof_address + 1]) * ELASTICITY
            self.physics.data.xpos[puck_id][1] = min_y + puck_radius / 1000 + nudge
        elif puck_pos[1] + puck_radius / 1000 >= max_y:
            self.physics.data.qvel[dof_address + 1] = -abs(self.physics.data.qvel[dof_address + 1]) * ELASTICITY
            self.physics.data.xpos[puck_id][1] = max_y - puck_radius / 1000 - nudge

        self.physics.data.qpos[self.physics.model.jnt_qposadr[puck_id] + 2] = 0.02

    def handle_paddle_collisions(self, paddle_name):
        if self.check_collision(paddle_name):
            puck_pos = np.array(self.get_position())
            paddle_pos = np.array(self.physics.get_position(paddle_name))

            puck_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
            self.physics.data.qpos[self.physics.model.jnt_qposadr[puck_id] + 2] = 0.02

            collision_vector = puck_pos - paddle_pos
            collision_vector /= np.linalg.norm(collision_vector)

            puck_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
            dof_address = self.physics.model.body_dofadr[puck_id]

            self.physics.data.qvel[dof_address] = -self.physics.data.qvel[dof_address] * ELASTICITY
            self.physics.data.qvel[dof_address + 1] = -self.physics.data.qvel[dof_address + 1] * ELASTICITY

            bounce_force = collision_vector * 0.3
            self.apply_force(bounce_force)

        paddle_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, paddle_name)
        self.physics.data.qpos[self.physics.model.jnt_qposadr[paddle_id] + 2] = 0.01

