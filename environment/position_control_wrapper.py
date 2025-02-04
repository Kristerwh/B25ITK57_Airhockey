import mujoco
import numpy as np
from collections import deque


class PositionControl:
    def __init__(self, p_gain=1.0, d_gain=0.1, i_gain=0.05, debug=False):
        """
        Position control system for paddles in the air hockey environment.

        Args:
            p_gain (float): Proportional controller gain.
            d_gain (float): Differential controller gain.
            i_gain (float): Integral controller gain.
            debug (bool): If True, logs the controller performance.
        """
        self.p_gain = np.array(p_gain)
        self.d_gain = np.array(d_gain)
        self.i_gain = np.array(i_gain)
        self.debug = debug

        # Controller state
        self.prev_pos = None
        self.prev_vel = None
        self.i_error = None

        if self.debug:
            self.controller_record = deque(maxlen=1000)

    def reset(self):
        """ Resets controller state """
        self.prev_pos = None
        self.prev_vel = None
        self.i_error = None
        if self.debug:
            self.controller_record.clear()

    def enforce_limits(self, position, velocity, joint_limits):
        """ Ensures paddles stay within allowed table boundaries. """
        position = np.clip(position, joint_limits[0], joint_limits[1])
        velocity = np.clip(velocity, -5.0, 5.0)
        return position, velocity

    def control(self, desired_pos, desired_vel, current_pos, current_vel):
        """
        Simple PD controller for paddle movement.

        Args:
            desired_pos (np.array): Target position.
            desired_vel (np.array): Target velocity.
            current_pos (np.array): Current paddle position.
            current_vel (np.array): Current paddle velocity.

        Returns:
            np.array: Computed control signal.
        """
        error = desired_pos - current_pos
        d_error = desired_vel - current_vel

        control_signal = self.p_gain * error + self.d_gain * d_error

        #Ensure movement constraints
        controlled_pos, controlled_vel = self.enforce_limits(desired_pos, desired_vel, [[-1.0, 1.0], [-0.5, 0.5]])

        if self.debug:
            self.controller_record.append([desired_pos, current_pos, desired_vel, current_vel])

        return controlled_pos, controlled_vel

    #does nothing yet, but will come handy when we start using joins with robot arms etc
    def update_joint_positions(self):
        pass
