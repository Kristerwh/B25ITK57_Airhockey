import copy
import numpy as np


class Constraints:
    TABLE_BOUNDS = [[-1.0, 1.0], [-0.5, 0.5]]
    PUCK_SPEED_LIMIT = 5.0
    FRICTION = 0.98
    BOUNCE_FACTOR = 0.9

    @staticmethod
    def enforce_table_bounds(position):
        if not isinstance(position, (list, np.ndarray)) or len(position) < 2:
            raise ValueError(f"Invalid position format: {position}. Expected [x, y] or [x, y, z].")

        position[0] = max(Constraints.TABLE_BOUNDS[0][0], min(Constraints.TABLE_BOUNDS[0][1], position[0]))
        position[1] = max(Constraints.TABLE_BOUNDS[1][0], min(Constraints.TABLE_BOUNDS[1][1], position[1]))
        return position

    @staticmethod
    def enforce_paddle_bounds(position, paddle_side):
        if paddle_side == "left":
            position[0] = min(4, position[0])
        elif paddle_side == "right":
            position[0] = max(-4, position[0])
        return position

    @staticmethod
    def apply_physics_constraints(velocity):
        velocity *= Constraints.FRICTION
        velocity = np.clip(velocity, -Constraints.PUCK_SPEED_LIMIT, Constraints.PUCK_SPEED_LIMIT)
        return velocity