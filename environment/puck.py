import mujoco
import numpy as np

from constants import puck_radius, paddle_radius


class Puck:
    def __init__(self, physics, body_name):
        self.physics = physics
        self.body_name = body_name
        self.body_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        self.geom_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_GEOM, self.body_name)

    def get_position(self):
        return self.physics.get_position(self.body_name)

    #temporary function to reset the puck, we'll expand on this later with goals and scores and more stuff
    def puck_reset(self):
        self.physics.set_paddle_position(self.body_name, [0, 0])
        self.physics.data.qvel[
        self.physics.model.body_dofadr[self.body_id]:self.physics.model.body_dofadr[self.body_id] + 2] = [0, 0]

    #temporary scoring function, just prints out which wall puck touched
    def point_scored(self):
        puck_pos = self.get_position()
        if -1.25 < puck_pos[0] <= -1.25 and abs(puck_pos[1]) <= 0.2:
            return "puck touched left wall"
        elif 1.25 <= puck_pos[0] < 1.25 and abs(puck_pos[1]) <= 0.2:
            return "puck touched right wal"
        return None

    def apply_impulse(self, force):
        dof_address = self.physics.model.body_dofadr[self.body_id]
        velocity = self.physics.data.qvel[dof_address:dof_address + 2]

        impulse = np.array(force)
        impulse /= np.linalg.norm(impulse) if np.linalg.norm(impulse) > 0 else 1

        self.physics.data.qvel[dof_address:dof_address + 2] += impulse * 0.1

    def get_elasticity(self):
        if self.geom_id != -1:
            return self.physics.model.geom_solimp[self.geom_id][0]
        return 0.95

    def get_friction(self):
        if self.geom_id != -1:
            return np.mean(self.physics.model.geom_friction[self.geom_id])
        return 0.3

    def apply_friction(self):
        friction = self.get_friction()
        dof_address = self.physics.model.body_dofadr[self.body_id]
        velocity = self.physics.data.qvel[dof_address:dof_address + 2]
        friction_factor = np.clip(1 - friction, 1.01, 2)
        self.physics.data.qvel[dof_address:dof_address + 2] *= friction_factor

    def check_collision(self, other_body_name):
        puck_pos = np.array(self.get_position())
        other_pos = np.array(self.physics.get_position(other_body_name))
        distance = np.linalg.norm(puck_pos - other_pos)
        return distance <= (puck_radius + paddle_radius)

    def handle_wall_collisions(self):
        elasticity = self.get_elasticity()
        for wall in ["wall_left", "wall_right", "wall_top", "wall_bottom"]:
            if self.physics.check_collision(self.body_name, wall):
                dof_address = self.physics.model.body_dofadr[self.body_id]
                velocity = self.physics.data.qvel[dof_address:dof_address + 1.0]

                if "left" in wall or "right" in wall:
                    velocity[0] = -velocity[0] * elasticity
                elif "top" in wall or "bottom" in wall:
                    velocity[1] = -velocity[1] * elasticity

                self.physics.data.qvel[dof_address:dof_address + 1.0] = velocity

    def handle_paddle_collisions(self, paddle_name):
        if self.physics.check_collision(paddle_name):
            puck_pos = np.array(self.get_position())
            paddle_pos = np.array(self.physics.get_position(paddle_name))

            collision_vector = puck_pos - paddle_pos
            if np.linalg.norm(collision_vector) != 0:
                collision_vector /= np.linalg.norm(collision_vector)
            self.apply_impulse(collision_vector * 1.5)

