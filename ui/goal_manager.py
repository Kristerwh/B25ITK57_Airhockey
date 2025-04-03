import mujoco
import numpy as np
import time

class GoalManager:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.goal_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_left")
        self.goal_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_right")

        self.goal_threshold = 0.03
        self.score_player = 0
        self.score_ai = 0

        self.reset_start_time = None
        self.reset_duration = 3
        self.in_reset = False

        self.reset_positions = {
            "puck": np.array([0.0, 0.0]),
            "player": np.array([-0.8, 0.0]),
            "ai": np.array([0.8, 0.0])
        }

    def check_goal(self):
        if self.in_reset:
            return None

        puck_x, puck_y = self.data.qpos[0], self.data.qpos[1]

        #Left goal(red side)
        left = self.model.site_pos[self.goal_left_id]
        if (
                left[0] - 0.01 <= puck_x <= left[0] + 0.01 and
                left[1] - 0.122 <= puck_y <= left[1] + 0.122
        ):
            self.score_ai += 1
            self._start_reset()
            return "AI +1"

        #Right goal(blue side)
        right = self.model.site_pos[self.goal_right_id]
        if (
                right[0] - 0.01 <= puck_x <= right[0] + 0.01 and
                right[1] - 0.122 <= puck_y <= right[1] + 0.122
        ):
            self.score_player += 1
            self._start_reset()
            return "PLAYER +1"

        return None

    def _start_reset(self):
        self.reset_start_time = time.time()
        self.in_reset = True

    def maybe_apply_reset(self):
        """Call this every frame. Applies reset if the time has passed."""
        if self.in_reset and time.time() - self.reset_start_time >= self.reset_duration:
            self._apply_reset()
            self.in_reset = False

    def _apply_reset(self):
        self.data.qpos[0:2] = self.reset_positions["puck"]
        self.data.qvel[0:2] = [0, 0]

        self.data.qpos[3:5] = self.reset_positions["player"]
        self.data.qvel[3:5] = [0, 0]

        self.data.qpos[5:7] = self.reset_positions["ai"]
        self.data.qvel[5:7] = [0, 0]

    def get_score(self):
        return self.score_player, self.score_ai

    def is_resetting(self):
        return self.in_reset
