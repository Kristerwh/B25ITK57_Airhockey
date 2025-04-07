import mujoco
import numpy as np
import time

class GoalManager:
    def __init__(self, model, data, table_length=2.0, table_width=1.0):
        self.model = model
        self.data = data

        self.table_length = table_length
        self.table_width = table_width
        self.goal_zone_y = 0.15

        self.goal_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_left")
        self.goal_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_right")

        self.goal_threshold = 0.03
        self.score_player = 0
        self.score_ai = 0

        self.reset_start_time = None
        self.reset_duration = 3
        self.in_reset = False
        self.goal_scored = False

        self.reset_positions = {
            "puck": np.array([0.0, 0.0]),
            "player": np.array([-0.8, 0.0]),
            "ai": np.array([0.8, 0.0])
        }

    def check_goal(self):
        if self.in_reset or self.goal_scored:
            return None

        puck_x, puck_y = self.data.qpos[0], self.data.qpos[1]
        puck_vel = np.linalg.norm(self.data.qvel[0:2])

        #Goal area bounds
        goal_x_threshold = 1.015
        goal_y_range = 0.15

        if puck_x > goal_x_threshold and abs(puck_y) < goal_y_range:
            self.score_player += 1
            self.goal_scored = True
            self._start_reset()
            return "PLAYER +1"

        #Check left goal (AI scores)
        if puck_x < -goal_x_threshold and abs(puck_y) < goal_y_range:
            self.score_ai += 1
            self.goal_scored = True
            self._start_reset()
            return "AI +1"

        if puck_vel > 100:
            print("Absorbing — abnormal speed")
            self.goal_scored = True
            self._start_reset()
            return "Reset (speed)"

        return None

    def _start_reset(self):
        self._trigger_slow_motion()
        self._run_countdown_overlay()
        self.reset_start_time = time.time()
        self.in_reset = True

    def _trigger_slow_motion(self):
        if hasattr(self.model.opt, 'timestep'):
            self.original_timestep = self.model.opt.timestep
            self.model.opt.timestep *= 0.2

    def _restore_speed(self):
        if hasattr(self.model.opt, 'timestep'):
            self.model.opt.timestep = self.original_timestep

    def _run_countdown_overlay(self):
        for i in range(3, 0, -1):
            print(f"Resetting in {i}...")
            time.sleep(1)

    def maybe_apply_reset(self):
        if self.in_reset and time.time() - self.reset_start_time >= self.reset_duration:
            self._apply_reset()
            self._restore_speed()
            self.in_reset = False
            self.goal_scored = False

    def _apply_reset(self):
        self.data.qpos[0:2] = self.reset_positions["puck"]
        self.data.qvel[0:2] = [0.0, 0.0]