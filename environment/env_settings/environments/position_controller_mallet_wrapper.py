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
        self.left_mallet_joint_ids = ["left_mallet_x_motor", "left_mallet_y_motor"]
        self.right_mallet_joint_ids = ["right_mallet_x_motor", "right_mallet_y_motor"]
        self.prev_vel = np.zeros(len(self.left_mallet_joint_ids) + len(self.right_mallet_joint_ids))

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
        max_speed = 200.0  # Adjust based on your simulation limits
        action = np.clip(action, -max_speed, max_speed)

        if self.debug:
            self.controller_record.append(action)

        return action

    def reset(self):
        """
        Reset the mallet control.
        """
        self.prev_vel = np.zeros(len(self.left_mallet_joint_ids) + len(self.right_mallet_joint_ids))
        if self.debug:
            self.controller_record.clear()

#for ppo agent
def apply_action_ppo(action, max_speed=100.0):
    """ PPO-safe motor velocity clamp """
    return np.clip(action, -max_speed, max_speed)