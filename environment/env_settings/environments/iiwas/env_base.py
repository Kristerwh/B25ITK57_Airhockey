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
        observation_spec = [("puck_pos", "puck", ObservationType.BODY_POS),
                            ("puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                            ("puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                            ("paddle_left_pos", "paddle_left",ObservationType.BODY_POS),
                            ("paddle_left_x_vel", "paddle_left_x", ObservationType.JOINT_VEL),
                            ("paddle_left_y_vel", "paddle_left_y", ObservationType.JOINT_VEL),
                            ]

        additional_data = observation_spec.copy()

        collision_spec = [("puck", ["puck"]),
                          ("rim", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r", "rim_left", "rim_right"]),
                          ("rim_short_sides", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"])]

        if self.n_agents == 1:
            scene = os.path.join(os.path.dirname(os.path.abspath(env_path)), "single.xml")
            action_spec += ["left_mallet_x_motor", "left_mallet_y_motor"]
            collision_spec += [
                ("paddle_left", ["puck", "rim_left", "rim_right", "rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"])]

        else:  # Two-player mode
            scene = os.path.join(os.path.dirname(os.path.abspath(env_path)), "two_player.xml")  # Load updated XML
            action_spec += ["left_mallet_x_motor", "left_mallet_y_motor",
                            "right_mallet_x_motor", "right_mallet_y_motor"]
            additional_data += [("paddle_right_x_pos", "paddle_right_x", ObservationType.JOINT_POS),
                                ("paddle_right_y_pos", "paddle_right_y", ObservationType.JOINT_POS),
                                ("paddle_right_x_vel", "paddle_right_x", ObservationType.JOINT_VEL),
                                ("paddle_right_y_vel", "paddle_right_y", ObservationType.JOINT_VEL)]
            collision_spec += [("paddle_left",
                                ["puck", "rim_left", "rim_right", "rim_home_l", "rim_home_r", "rim_away_l",
                                 "rim_away_r"]),
                               ("paddle_right",
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

    def reward(self, obs, action, next_obs, absorbing):
        puck_pos, puck_vel = self.get_puck(obs)
        mallet_pos, mallet_vel = self.get_mallet(obs)
        # print(f"Mallet pos: {mallet_pos}, Mallet vel: {mallet_vel}")
        reward = 0

        # Puck on AI side
        if puck_pos[0] <= 0:
            # Penalize distance to puck
            dist = np.linalg.norm(puck_pos - mallet_pos)
            reward -= min(dist, 1.0) * 0.05

            # Bonus for hitting the puck
            if self.is_colliding(next_obs, 'puck', 'paddle_left'):
                reward += 10.0

            # Opponent goal
            if absorbing:
                print("Opponent Scores!")
                reward -= 50

            # Bonus if moving toward the puck
            dx, dy = puck_pos - mallet_pos
            mvx, mvy = mallet_vel
            if (dx * mvx >= 0) and (dy * mvy >= 0):
                reward += 0.05

        # Mallet on opponents side
        if mallet_pos[0] > 0:
            return -1

        # Puck on opponents side
        if puck_pos[0] > 0:
            reward += 0.1
            if absorbing:
                reward += 50
                print("AI Scores!")
        return reward

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

        if np.any(np.abs(puck_pos) > boundary) or np.linalg.norm(puck_vel) > 100:
            print("Absorbing")
            return True
        return False

    def get_puck(self, obs):
        """
        Getting the puck properties from the observations

        Returns:
            ([pos_x, pos_y], [vel_x, vel_y])
        """
        puck_pos = self.obs_helper.get_from_obs(obs, "puck_pos")[:2]  # from BODY_POS: x, y
        puck_vel = np.concatenate([
            self.obs_helper.get_from_obs(obs, "puck_x_vel"),
            self.obs_helper.get_from_obs(obs, "puck_y_vel")
        ])
        return puck_pos, puck_vel

    def get_mallet(self, obs):
        mallet_pos = self.obs_helper.get_from_obs(obs, "paddle_left_pos")[:2]
        mallet_vel = np.concatenate([
            self.obs_helper.get_from_obs(obs, "paddle_left_x_vel"),
            self.obs_helper.get_from_obs(obs, "paddle_left_y_vel")
        ])
        return mallet_pos, mallet_vel

    def is_colliding(self, obs, obj1='puck', obj2='paddle_left'):
        radius = {
            'puck': self.env_info['puck']['radius'],
            'paddle_left': self.env_info['mallet']['radius'],
        }

        if obj1 == 'puck':
            pos1, _ = self.get_puck(obs)
        elif obj1 == 'paddle_left':
            pos1, _ = self.get_mallet(obs)
        else:
            raise ValueError(f"Unsupported object: {obj1}")

        if obj2 == 'puck':
            pos2, _ = self.get_puck(obs)
        elif obj2 == 'paddle_left':
            pos2, _ = self.get_mallet(obs)
        else:
            raise ValueError(f"Unsupported object: {obj2}")

        distance = np.linalg.norm(pos1 - pos2)
        min_distance = radius[obj1] + radius[obj2]

        return distance <= min_distance

    def _randomize_puck_position(self):
        # Sample random position
        min_dist = 0.5
        while True:
            x = np.random.uniform(*(-0.55, 0.05))
            y = np.random.uniform(*(-0.4, 0.4))
            mx = np.random.uniform(*(-0.45, 0.1)) + 0.35
            my = np.random.uniform(*(-0.4, 0.4))
            distance = np.linalg.norm([x - mx, y - my])
            if distance >= min_dist:
                break  # valid sample


        # Set the puck position directly in MuJoCo
        puck_body_id = self._model.body("puck").id
        self._data.qpos[self._model.jnt("puck_x").qposadr] = x
        self._data.qpos[self._model.jnt("puck_y").qposadr] = y
        mallet_body_id = self._model.body("paddle_left").id
        self._data.qpos[self._model.jnt("paddle_left_x").qposadr] = mx
        self._data.qpos[self._model.jnt("paddle_left_y").qposadr] = my

    def reset(self, obs=None):
        super().reset(obs)  # This will set up model/data/obs_helper and reset MuJoCo
        self._randomize_puck_position()
        mujoco.mj_forward(self._model, self._data)

        # Recreate observation from updated state
        self._obs = self._create_observation(self.obs_helper._build_obs(self._data))
        return self._obs
