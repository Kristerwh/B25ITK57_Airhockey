import mujoco
import numpy as np
from collections import deque

class MalletControl:
    def __init__(self, env_info=None, debug=False, *args, **kwargs):
        """
        Simple velocity-based control for the mallet in an air hockey environment.

        Args:
            env_info (dict): Environment metadata (contains table size, puck position, etc.)
            debug (bool): If True, logs actions.
        """
        self.env_info = env_info or {}
        self.debug = debug

        # Define mallet action space
        self.actuator_joint_ids = ["mallet_x_motor", "mallet_y_motor"]
        self.prev_vel = np.zeros(len(self.actuator_joint_ids))

        if self.debug:
            self.controller_record = deque(maxlen=1000)

    def apply_action(self, action):
        """
        Apply the given velocity command to the mallet.

        Args:
            action (np.array): A 2D array [x_velocity, y_velocity]
        Returns:
            control_signal (np.array): Processed velocity commands
        """
        # Ensure the action is within the allowed velocity range
        max_speed = 2.0  # Adjust based on your simulation limits
        action = np.clip(action, -max_speed, max_speed)

        if self.debug:
            self.controller_record.append(action)

        return action

    def reset(self):
        """
        Reset the mallet control.
        """
        self.prev_vel = np.zeros(len(self.actuator_joint_ids))
        if self.debug:
            self.controller_record.clear()
