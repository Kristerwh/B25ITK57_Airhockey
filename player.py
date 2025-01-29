import mujoco

from constants import WIDTH, HEIGHT, paddle_move_range, paddle_radius


class Player:
    def __init__(self, physics, body_name, start_position, ai_player=False):
        self.physics = physics
        self.body_name = body_name
        self.start_position = start_position
        self.ai_player = ai_player
        self.velocity = [0, 0]

    def controls(self, direction):
        current_pos = self.physics.get_position(self.body_name)

        min_x, max_x = -0.6 + paddle_radius / 1000, 0.6 - paddle_radius / 1000
        min_y, max_y = -0.4 + paddle_radius / 1000, 0.4 - paddle_radius / 1000

        acceleration = 0.02
        deceleration = 0.05

        if direction == "up" and current_pos[1] < max_y:
            self.velocity[1] += acceleration
        elif direction == "down" and current_pos[1] > min_y:
            self.velocity[1] -= acceleration
        else:
            self.velocity[1] *= (1 - deceleration)

        if direction == "left" and current_pos[0] > min_x:
            self.velocity[0] -= acceleration
        elif direction == "right" and current_pos[0] < max_x:
            self.velocity[0] += acceleration
        else:
            self.velocity[0] *= (1 - deceleration)

        self.physics.apply_force(self.body_name, self.velocity)

        # ##for controls for ''first person'' perspective, commented out for now, might be needed later
        # if direction == "left" and current_pos[1] < max_y:
        #     self.physics.apply_force(self.body_name, [0, force_magnitude])
        # elif direction == "right" and current_pos[1] > min_y:
        #     self.physics.apply_force(self.body_name, [0, -force_magnitude])
        # elif direction == "down" and current_pos[0] > min_x:
        #     self.physics.apply_force(self.body_name, [-force_magnitude, 0])
        # elif direction == "up" and current_pos[0] < max_x:
        #     self.physics.apply_force(self.body_name, [force_magnitude, 0])

    def stop(self):
        self.physics.apply_force(self.body_name, [0, 0])

        paddle_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)

        if paddle_id < len(self.physics.model.jnt_qposadr):
            joint_index = self.physics.model.jnt_qposadr[paddle_id]
            if joint_index + 2 < len(self.physics.data.qpos):
                self.physics.data.qpos[joint_index + 2] = 0.02