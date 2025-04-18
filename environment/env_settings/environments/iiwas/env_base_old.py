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

            # We are not using robot arms, therefore there is no need to keep it in

            # action_spec += ["iiwa_1/joint_1", "iiwa_1/joint_2", "iiwa_1/joint_3", "iiwa_1/joint_4", "iiwa_1/joint_5",
            #                 "iiwa_1/joint_6", "iiwa_1/joint_7"]
            # observation_spec += [("robot_1/joint_1_pos", "iiwa_1/joint_1", ObservationType.JOINT_POS),
            #                      ("robot_1/joint_2_pos", "iiwa_1/joint_2", ObservationType.JOINT_POS),
            #                      ("robot_1/joint_3_pos", "iiwa_1/joint_3", ObservationType.JOINT_POS),
            #                      ("robot_1/joint_4_pos", "iiwa_1/joint_4", ObservationType.JOINT_POS),
            #                      ("robot_1/joint_5_pos", "iiwa_1/joint_5", ObservationType.JOINT_POS),
            #                      ("robot_1/joint_6_pos", "iiwa_1/joint_6", ObservationType.JOINT_POS),
            #                      ("robot_1/joint_7_pos", "iiwa_1/joint_7", ObservationType.JOINT_POS),
            #                      ("robot_1/joint_1_vel", "iiwa_1/joint_1", ObservationType.JOINT_VEL),
            #                      ("robot_1/joint_2_vel", "iiwa_1/joint_2", ObservationType.JOINT_VEL),
            #                      ("robot_1/joint_3_vel", "iiwa_1/joint_3", ObservationType.JOINT_VEL),
            #                      ("robot_1/joint_4_vel", "iiwa_1/joint_4", ObservationType.JOINT_VEL),
            #                      ("robot_1/joint_5_vel", "iiwa_1/joint_5", ObservationType.JOINT_VEL),
            #                      ("robot_1/joint_6_vel", "iiwa_1/joint_6", ObservationType.JOINT_VEL),
            #                      ("robot_1/joint_7_vel", "iiwa_1/joint_7", ObservationType.JOINT_VEL)]
            # additional_data += [("robot_1/joint_8_pos", "iiwa_1/striker_joint_1", ObservationType.JOINT_POS),
            #                     ("robot_1/joint_9_pos", "iiwa_1/striker_joint_2", ObservationType.JOINT_POS),
            #                     ("robot_1/joint_8_vel", "iiwa_1/striker_joint_1", ObservationType.JOINT_VEL),
            #                     ("robot_1/joint_9_vel", "iiwa_1/striker_joint_2", ObservationType.JOINT_VEL),
            #                     ("robot_1/ee_pos", "iiwa_1/striker_mallet", ObservationType.BODY_POS),
            #                     ("robot_1/ee_vel", "iiwa_1/striker_mallet", ObservationType.BODY_VEL),
            #                     ("robot_1/rod_rot", "iiwa_1/striker_joint_link", ObservationType.BODY_ROT)]
            # collision_spec += [("robot_1/ee", ["iiwa_1/ee"])]
            
            # OUR VERSION
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

            # We do not need a second agent as we will be using Scripted AI for the second player.
            # if self.n_agents == 2:
            #     scene = os.path.join(os.path.dirname(os.path.abspath(env_path)), "double.xml")
            #     observation_spec += [("robot_1/opponent_ee_pos", "iiwa_2/striker_joint_link", ObservationType.BODY_POS)]
            #     action_spec += ["iiwa_2/joint_1", "iiwa_2/joint_2", "iiwa_2/joint_3", "iiwa_2/joint_4",
            #                     "iiwa_2/joint_5",
            #                     "iiwa_2/joint_6", "iiwa_2/joint_7"]
            #     observation_spec += [("robot_2/puck_x_pos", "puck_x", ObservationType.JOINT_POS),
            #                          ("robot_2/puck_y_pos", "puck_y", ObservationType.JOINT_POS),
            #                          ("robot_2/puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
            #                          ("robot_2/puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
            #                          ("robot_2/puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
            #                          ("robot_2/puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL),
            #                          ("robot_2/joint_1_pos", "iiwa_2/joint_1", ObservationType.JOINT_POS),
            #                          ("robot_2/joint_2_pos", "iiwa_2/joint_2", ObservationType.JOINT_POS),
            #                          ("robot_2/joint_3_pos", "iiwa_2/joint_3", ObservationType.JOINT_POS),
            #                          ("robot_2/joint_4_pos", "iiwa_2/joint_4", ObservationType.JOINT_POS),
            #                          ("robot_2/joint_5_pos", "iiwa_2/joint_5", ObservationType.JOINT_POS),
            #                          ("robot_2/joint_6_pos", "iiwa_2/joint_6", ObservationType.JOINT_POS),
            #                          ("robot_2/joint_7_pos", "iiwa_2/joint_7", ObservationType.JOINT_POS),
            #                          ("robot_2/joint_1_vel", "iiwa_2/joint_1", ObservationType.JOINT_VEL),
            #                          ("robot_2/joint_2_vel", "iiwa_2/joint_2", ObservationType.JOINT_VEL),
            #                          ("robot_2/joint_3_vel", "iiwa_2/joint_3", ObservationType.JOINT_VEL),
            #                          ("robot_2/joint_4_vel", "iiwa_2/joint_4", ObservationType.JOINT_VEL),
            #                          ("robot_2/joint_5_vel", "iiwa_2/joint_5", ObservationType.JOINT_VEL),
            #                          ("robot_2/joint_6_vel", "iiwa_2/joint_6", ObservationType.JOINT_VEL),
            #                          ("robot_2/joint_7_vel", "iiwa_2/joint_7", ObservationType.JOINT_VEL)]
            #     observation_spec += [("robot_2/opponent_ee_pos", "iiwa_1/striker_joint_link", ObservationType.BODY_POS)]
            #     additional_data += [("robot_2/joint_8_pos", "iiwa_2/striker_joint_1", ObservationType.JOINT_POS),
            #                         ("robot_2/joint_9_pos", "iiwa_2/striker_joint_2", ObservationType.JOINT_POS),
            #                         ("robot_2/joint_8_vel", "iiwa_2/striker_joint_1", ObservationType.JOINT_VEL),
            #                         ("robot_2/joint_9_vel", "iiwa_2/striker_joint_2", ObservationType.JOINT_VEL),
            #                         ("robot_2/ee_pos", "iiwa_2/striker_mallet", ObservationType.BODY_POS),
            #                         ("robot_2/ee_vel", "iiwa_2/striker_mallet", ObservationType.BODY_VEL),
            #                         ("robot_2/rod_rot", "iiwa_2/striker_joint_link", ObservationType.BODY_ROT)]
            #     collision_spec += [("robot_2/ee", ["iiwa_2/ee"])]
        else:
            raise ValueError('n_agents should be 1')

        self.env_info = dict()
        self.env_info['table'] = {"length": 1.948, "width": 1.038, "goal_width": 0.25}
        self.env_info['puck'] = {"radius": 0.03165}
        self.env_info['mallet'] = {"radius": 0.04815}
        self.env_info['n_agents'] = self.n_agents
        
        # ROBOT
        # self.env_info['robot'] = {
        #     "n_joints": 7,
        #     "ee_desired_height": 0.1645,
        #     "joint_vel_limit": np.array([[-85, -85, -100, -75, -130, -135, -135],
        #                                  [85, 85, 100, 75, 130, 135, 135]]) / 180. * np.pi,
        #     "joint_acc_limit": np.array([[-85, -85, -100, -75, -130, -135, -135],
        #                                  [85, 85, 100, 75, 130, 135, 135]]) / 180. * np.pi * 10,
        #     "base_frame": [],
        #     "universal_height": 0.0645,
        #     "control_frequency": 50,
        # }

        self.env_info['puck_pos_ids'] = [0, 1, 2]
        self.env_info['puck_vel_ids'] = [3, 4, 5]
        self.env_info['opponent_ee_ids'] = []
        # self.env_info['joint_pos_ids'] = [6, 7, 8, 9, 10, 11, 12]
        # self.env_info['joint_vel_ids'] = [13, 14, 15, 16, 17, 18, 19]
        # if self.n_agents == 2:
        #     self.env_info['opponent_ee_ids'] = [20, 21, 22]
        # else:
        #     self.env_info['opponent_ee_ids'] = []

        # max_joint_vel = ([np.inf] * 3 + list(self.env_info["robot"]["joint_vel_limit"][1, :7])) * self.n_agents

        super().__init__(scene, action_spec, observation_spec, gamma, horizon, timestep, n_substeps,
                         n_intermediate_steps, additional_data, collision_spec, **viewer_params)
        # Removed max_joint_vel from init

        # ROBOT XML
        # Construct the mujoco model at origin
        # robot_model = mujoco.MjModel.from_xml_path(
        #     os.path.join(os.path.dirname(os.path.abspath(env_path)), "iiwa_only.xml"))
        # robot_model.body('iiwa_1/base').pos = np.zeros(3)
        # robot_data = mujoco.MjData(robot_model)

        # Add env_info that requires mujoco models
        self.env_info['dt'] = self.dt
        # ROBOT
        # self.env_info["robot"]["joint_pos_limit"] = np.array(
        #     [self._model.joint(f"iiwa_1/joint_{i + 1}").range for i in range(7)]).T
        # self.env_info["robot"]["robot_model"] = robot_model
        # self.env_info["robot"]["robot_data"] = robot_data
        self.env_info["rl_info"] = self.info

        # ROBOT
        # frame_T = np.eye(4)
        # temp = np.zeros((9, 1))
        # mujoco.mju_quat2Mat(temp, self._model.body("iiwa_1/base").quat)
        # frame_T[:3, :3] = temp.reshape(3, 3)
        # frame_T[:3, 3] = self._model.body("iiwa_1/base").pos
        # self.env_info['robot']['base_frame'].append(frame_T.copy())

        # We are only using 1 agent
        # if self.n_agents == 2:
        #     mujoco.mju_quat2Mat(temp, self._model.body("iiwa_2/base").quat)
        #     frame_T[:3, :3] = temp.reshape(3, 3)
        #     frame_T[:3, 3] = self._model.body("iiwa_2/base").pos
        #     self.env_info['robot']['base_frame'].append(frame_T.copy())

        # Ids of the joint, which are controller by the action space
        # self.actuator_joint_ids = [self._model.joint(name).id for name in action_spec]

        # self.universal_joint_plugin = UniversalJointPlugin(self._model, self._data, self.env_info)

    # def _modify_mdp_info(self, mdp_info):
    #     obs_low = np.array([0, -1, -np.pi, -20., -20., -100,
    #                         *np.array([self._model.joint(f"iiwa_1/joint_{i + 1}").range[0]
    #                                    for i in range(self.env_info['robot']['n_joints'])]),
    #                         *self.env_info['robot']['joint_vel_limit'][0]])
    #     obs_high = np.array([3.02, 1, np.pi, 20., 20., 100,
    #                          *np.array([self._model.joint(f"iiwa_1/joint_{i + 1}").range[1]
    #                                     for i in range(self.env_info['robot']['n_joints'])]),
    #                          *self.env_info['robot']['joint_vel_limit'][1]])
        # if self.n_agents == 2:
        #     obs_low = np.concatenate([obs_low, [1.5, -1.5, -1.5]])
        #     obs_high = np.concatenate([obs_high, [4.5, 1.5, 1.5]])
        # mdp_info.observation_space = Box(obs_low, obs_high)
        # return mdp_info

    # TODO
    # MODIFIED MUST BE LOOKED INTO
    def _modify_mdp_info(self, mdp_info):
        obs_low = np.array([-1.0, -0.5, -5.0, -5.0, -1.0, -0.5])  # Min values for mallet & puck
        obs_high = np.array([1.0, 0.5, 5.0, 5.0, 1.0, 0.5])  # Max values for mallet & puck
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info
    
    # ROBOT
    # def _simulation_pre_step(self):
    #     self.universal_joint_plugin.update()

    def is_absorbing(self, obs):
        boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2
        puck_pos, puck_vel = self.get_puck(obs)

        if np.any(np.abs(puck_pos[:2]) > boundary) or np.linalg.norm(puck_vel) > 100:
            return True
        return False

    # ROBOT PUCK HANDLING
    # @staticmethod
    # def _puck_2d_in_robot_frame(puck_in, robot_frame, type='pose'):
    #     if type == 'pose':
    #         puck_w = np.eye(4)
    #         puck_w[:2, 3] = puck_in[:2]
    #         puck_w[:3, :3] = R.from_euler("xyz", [0., 0., puck_in[2]]).as_matrix()

    #         puck_r = np.linalg.inv(robot_frame) @ puck_w
    #         puck_out = np.concatenate([puck_r[:2, 3],
    #                                    R.from_matrix(puck_r[:3, :3]).as_euler('xyz')[2:3]])

    #     if type == 'vel':
    #         rot_mat = robot_frame[:3, :3]

    #         vel_lin = np.array([*puck_in[:2], 0])
    #         vel_ang = np.array([0., 0., puck_in[2]])

    #         vel_lin_r = rot_mat.T @ vel_lin
    #         vel_ang_r = rot_mat.T @ vel_ang

    #         puck_out = np.concatenate([vel_lin_r[:2], vel_ang_r[2:3]])
    #     return puck_out

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

    # def get_ee(self):
    #     raise NotImplementedError

    # def get_joints(self, obs):
    #     raise NotImplementedError
