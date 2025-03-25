import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from environment.env_settings.environments.data.iiwas import __file__ as env_path
from environment.env_settings.utils.universal_joint_plugin import UniversalJointPlugin
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.utils.spaces import Box


"""
    Abstract class for all AirHockey Environments.

"""
class AirHockeyBase(MuJoCo):
    def __init__(self, gamma=0.99, horizon=500, timestep=1 / 1000., n_intermediate_steps=20, n_substeps=1,
                 n_agents=1, viewer_params={}):

        """
            Constructor.

            Args:
                n_agents (int): Number of agents (1 or 2)
        """
        self.n_agents = n_agents
        if self.n_agents not in [1, 2]:
            raise ValueError('n_agents must be either 1 or 2')

        action_spec = []
        observation_spec = [("puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                            ("puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                            ("puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
                            ("puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                            ("puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                            ("puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL)]

        additional_data = observation_spec.copy()

        collision_spec = [("puck", ["puck"]),
                          ("rim", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r", "rim_left", "rim_right"]),
                          ("rim_short_sides", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"])]

        if self.n_agents == 1:
            scene = os.path.join(os.path.dirname(os.path.abspath(env_path)), "single.xml")
            action_spec += ["left_mallet_x_motor", "left_mallet_y_motor"]
            additional_data += [("left_mallet_x_pos", "left_mallet_x", ObservationType.JOINT_POS),
                                ("left_mallet_y_pos", "left_mallet_y", ObservationType.JOINT_POS),
                                ("left_mallet_x_vel", "left_mallet_x", ObservationType.JOINT_VEL),
                                ("left_mallet_y_vel", "left_mallet_y", ObservationType.JOINT_VEL), ]
            collision_spec += [
                ("left_mallet", ["puck", "rim_left", "rim_right", "rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"])]

        else:  # Two-player mode
            scene = os.path.join(os.path.dirname(os.path.abspath(env_path)), "two_player.xml")  # Load updated XML
            action_spec += ["left_mallet_x_motor", "left_mallet_y_motor",
                            "right_mallet_x_motor", "right_mallet_y_motor"]
            additional_data += [("left_mallet_x_pos", "left_mallet_x", ObservationType.JOINT_POS),
                                ("left_mallet_y_pos", "left_mallet_y", ObservationType.JOINT_POS),
                                ("left_mallet_x_vel", "left_mallet_x", ObservationType.JOINT_VEL),
                                ("left_mallet_y_vel", "left_mallet_y", ObservationType.JOINT_VEL),
                                ("right_mallet_x_pos", "right_mallet_x", ObservationType.JOINT_POS),
                                ("right_mallet_y_pos", "right_mallet_y", ObservationType.JOINT_POS),
                                ("right_mallet_x_vel", "right_mallet_x", ObservationType.JOINT_VEL),
                                ("right_mallet_y_vel", "right_mallet_y", ObservationType.JOINT_VEL)]
            collision_spec += [("left_mallet",
                                ["puck", "rim_left", "rim_right", "rim_home_l", "rim_home_r", "rim_away_l",
                                 "rim_away_r"]),
                               ("right_mallet",
                                ["puck", "rim_left", "rim_right", "rim_home_l", "rim_home_r", "rim_away_l",
                                 "rim_away_r"])]

        self.env_info = dict()
        self.env_info['table'] = {"length": 1.948, "width": 1.038, "goal_width": 0.25}
        self.env_info['puck'] = {"radius": 0.03165}
        self.env_info['mallet'] = {"radius": 0.04815}
        self.env_info['n_agents'] = self.n_agents

        self.env_info['puck_pos_ids'] = [0, 1, 2]
        self.env_info['puck_vel_ids'] = [3, 4, 5]
        self.env_info['opponent_ee_ids'] = []

        super().__init__(scene, action_spec, observation_spec, gamma, horizon, timestep, n_substeps,
                         n_intermediate_steps, additional_data, collision_spec, **viewer_params)

        self.env_info['dt'] = self.dt
        self.env_info["rl_info"] = self.info

    # TODO
    # MODIFIED MUST BE LOOKED INTO
    def _modify_mdp_info(self, mdp_info):
        obs_low = np.array([-1.0, -0.5, -5.0, -5.0, -1.0, -0.5])  # Min values for mallet & puck
        obs_high = np.array([1.0, 0.5, 5.0, 5.0, 1.0, 0.5])  # Max values for mallet & puck
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info

    def is_absorbing(self, obs):
        boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2
        puck_pos, puck_vel = self.get_puck(obs)

        if np.any(np.abs(puck_pos[:2]) > boundary) or np.linalg.norm(puck_vel) > 100:
            return True
        return False


    def get_puck(self, obs):
        """
        Getting the puck properties from the observations
        Args:
            obs: The current observation

        Returns:
            ([pos_x, pos_y, yaw], [lin_vel_x, lin_vel_y, yaw_vel])

        """
        puck_pos = np.concatenate([self.obs_helper.get_from_obs(obs, "puck_x_pos"),
                                   self.obs_helper.get_from_obs(obs, "puck_y_pos"),
                                   self.obs_helper.get_from_obs(obs, "puck_yaw_pos")])
        puck_vel = np.concatenate([self.obs_helper.get_from_obs(obs, "puck_x_vel"),
                                   self.obs_helper.get_from_obs(obs, "puck_y_vel"),
                                   self.obs_helper.get_from_obs(obs, "puck_yaw_vel")])
        return puck_pos, puck_vel

    #---------------------------------rl testing------------------------------------------------------
    def reward(self, state, action, next_state, absorbing):
        print("next_state:", next_state)
        print("len(next_state):", len(next_state))

        puck_x, puck_y = next_state[0], next_state[1]
        puck_vx, puck_vy = next_state[3], next_state[4]
        reward = 0.0

        if puck_x > 1.3:
            reward += 1.0
            print("+1")
        elif puck_x < -1.3:
            reward -= 1.0
            print("-1")
        elif puck_x < -1.15:
            reward -= 0.5
            print("-0.5")

        if len(next_state) >= 10:
            mallet_x, mallet_y = next_state[6], next_state[7]
            mallet_vx, mallet_vy = next_state[8], next_state[9]

            dist_to_puck = np.linalg.norm([puck_x - mallet_x, puck_y - mallet_y])
            shaping = 2.0 * max(0, 1.0 - dist_to_puck * 5)
            reward += shaping
            print(f"Shaping reward (distance): {shaping:.3f}")

            speed = np.linalg.norm([mallet_vx, mallet_vy])
            if speed < 0.01:
                reward -= 0.5
                print("-0.5")

        print(f"Final reward: {reward:.3f}")
        return reward



