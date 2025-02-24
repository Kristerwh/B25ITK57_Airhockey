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
                n_agents (int, 1): number of agent to be used in the environment (one or two)
        """

        self.n_agents = n_agents

        action_spec = []
        observation_spec = [("puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                            ("puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                            ("puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
                            ("puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                            ("puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                            ("puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL)]

        additional_data = [("puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                           ("puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                           ("puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
                           ("puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                           ("puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                           ("puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL)]

        collision_spec = [("puck", ["puck"]),
                          ("rim", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r", "rim_left", "rim_right"]),
                          ("rim_short_sides", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"])]

        if 1 <= self.n_agents <= 2:
            scene = os.path.join(os.path.dirname(os.path.abspath(env_path)), "single.xml")
            action_spec += ["mallet_x_vel", "mallet_y_vel"]

            observation_spec += [("puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                                 ("puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                                 ("puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                                 ("puck_y_vel", "puck_y", ObservationType.JOINT_VEL),]
            
            additional_data += [("mallet_x_pos", "mallet_x", ObservationType.JOINT_POS),
                                ("mallet_y_pos", "mallet_y", ObservationType.JOINT_POS),
                                ("mallet_x_vel", "mallet_x", ObservationType.JOINT_VEL),
                                ("mallet_y_vel", "mallet_y", ObservationType.JOINT_VEL),]
            
            collision_spec += [("mallet", ["puck", "table_walls"])]
        else:
            raise ValueError('n_agents should be 1')

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