# File: airhockey_gym_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.env_settings.environments.iiwas.env_base import AirHockeyBase

class AirHockeyGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = AirHockeyBase()

        raw_obs = self.env.reset()
        extended_obs = self._get_obs(raw_obs)

        self.observation_space = spaces.Box(
            low=np.full_like(extended_obs, -1.0),
            high=np.full_like(extended_obs, 1.0),
            dtype=np.float32
        )

        act_dim = self.env.info.action_space.shape[0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

    def _get_obs(self, obs):
        try:
            mallet_pos = self.env._data.qpos[:2].copy()
            mallet_vel = self.env._data.qvel[:2].copy()
        except Exception as e:
            mallet_pos = np.array([0.0, 0.0])
            mallet_vel = np.array([0.0, 0.0])

        extended = np.concatenate([obs, mallet_pos, mallet_vel])
        return extended

    def reset(self, seed=None, options=None):
        raw_obs = self.env.reset()
        wrapped_obs = self._get_obs(raw_obs)
        return wrapped_obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        extended_obs = self._get_obs(obs)
        terminated = done
        truncated = False
        return extended_obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
