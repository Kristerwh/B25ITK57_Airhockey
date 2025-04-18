import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from environment.env_settings.environments.iiwas import AirHockeyBase
from environment.env_settings.utils.kinematics import inverse_kinematics


class AirHockeyDouble(AirHockeyBase):
    """
    Base class for two agents air hockey tasks.
    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}, **kwargs):

        super().__init__(gamma=gamma, horizon=horizon, n_agents=2, viewer_params=viewer_params, **kwargs)

        self._compute_init_state()

        self.filter_ratio = 0.274
        self.q_pos_prev = np.zeros(self.env_info["robot"]["n_joints"] * self.env_info["n_agents"])
        self.q_vel_prev = np.zeros(self.env_info["robot"]["n_joints"] * self.env_info["n_agents"])

    def _compute_init_state(self):
        init_state = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])

        success, self.init_state = inverse_kinematics(self.env_info['robot']['robot_model'],
                                                      self.env_info['robot']['robot_data'],
                                                      np.array([0.65, 0., 0.1645]),
                                                      R.from_euler('xyz', [0, 5 / 6 * np.pi, 0]).as_matrix(),
                                                      initial_q=init_state)

        assert success is True

    def get_ee(self, robot=1):
        """
        Getting the ee properties from the current internal state the selected robot. Can also be obtained via forward kinematics
        on the current joint position, this function exists to avoid redundant computations.
        Args:
            robot: ID of robot, either 1 or 2

        Returns: ([pos_x, pos_y, pos_z], [ang_vel_x, ang_vel_y, ang_vel_z, lin_vel_x, lin_vel_y, lin_vel_z])
        """
        ee_pos = self._read_data("robot_" + str(robot) + "/ee_pos")

        ee_vel = self._read_data("robot_" + str(robot) + "/ee_vel")

        return ee_pos, ee_vel

    def get_joints(self, obs, agent=None):
        """
        Get joint position and velocity of the robots
        Can choose the robot with agent = 1 / 2. If agent is None both are returned
        """
        if agent:
            q_pos = np.zeros(7)
            q_vel = np.zeros(7)
            for i in range(7):
                q_pos[i] = self.obs_helper.get_from_obs(obs, "robot_" + str(agent) + "/joint_" + str(i + 1) + "_pos")[0]
                q_vel[i] = self.obs_helper.get_from_obs(obs, "robot_" + str(agent) + "/joint_" + str(i + 1) + "_vel")[0]
        else:
            q_pos = np.zeros(14)
            q_vel = np.zeros(14)
            for i in range(7):
                q_pos[i] = self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i + 1) + "_pos")[0]
                q_vel[i] = self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i + 1) + "_vel")[0]

                q_pos[i + 7] = self.obs_helper.get_from_obs(obs, "robot_2/joint_" + str(i + 1) + "_pos")[0]
                q_vel[i + 7] = self.obs_helper.get_from_obs(obs, "robot_2/joint_" + str(i + 1) + "_vel")[0]

        return q_pos, q_vel

    def _create_observation(self, obs):
        # Filter the joint velocity
        q_pos, q_vel = self.get_joints(obs)
        q_vel_filter = self.filter_ratio * q_vel + (1 - self.filter_ratio) * self.q_vel_prev
        self.q_pos_prev = q_pos
        self.q_vel_prev = q_vel_filter

        for i in range(7):
            self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i + 1) + "_vel")[:] = q_vel_filter[i]
            self.obs_helper.get_from_obs(obs, "robot_2/joint_" + str(i + 1) + "_vel")[:] = q_vel_filter[i + 7]

        # Wrap puck's rotation angle to [-pi, pi)
        yaw_angle = self.obs_helper.get_from_obs(obs, "puck_yaw_pos")
        self.obs_helper.get_from_obs(obs, "puck_yaw_pos")[:] = (yaw_angle + np.pi) % (2 * np.pi) - np.pi
        return obs

    def _modify_observation(self, obs):
        new_obs = obs.copy()

        puck_pos, puck_vel = self.get_puck(new_obs)

        puck_pos_1 = self._puck_2d_in_robot_frame(puck_pos, self.env_info['robot']['base_frame'][0])
        puck_vel_1 = self._puck_2d_in_robot_frame(puck_vel, self.env_info['robot']['base_frame'][0], type='vel')

        self.obs_helper.get_from_obs(new_obs, "puck_x_pos")[:] = puck_pos_1[0]
        self.obs_helper.get_from_obs(new_obs, "puck_y_pos")[:] = puck_pos_1[1]
        self.obs_helper.get_from_obs(new_obs, "puck_yaw_pos")[:] = puck_pos_1[2]

        self.obs_helper.get_from_obs(new_obs, "puck_x_vel")[:] = puck_vel_1[0]
        self.obs_helper.get_from_obs(new_obs, "puck_y_vel")[:] = puck_vel_1[1]
        self.obs_helper.get_from_obs(new_obs, "puck_yaw_vel")[:] = puck_vel_1[2]

        opponent_pos_1 = self.obs_helper.get_from_obs(new_obs, 'robot_1/opponent_ee_pos')
        self.obs_helper.get_from_obs(new_obs, 'robot_1/opponent_ee_pos')[:] = \
            (np.linalg.inv(self.env_info['robot']['base_frame'][0]) @ np.concatenate([opponent_pos_1, [1]]))[:3]

        puck_pos_2 = self._puck_2d_in_robot_frame(puck_pos, self.env_info['robot']['base_frame'][1])
        puck_vel_2 = self._puck_2d_in_robot_frame(puck_vel, self.env_info['robot']['base_frame'][1], type='vel')

        self.obs_helper.get_from_obs(new_obs, "robot_2/puck_x_pos")[:] = puck_pos_2[0]
        self.obs_helper.get_from_obs(new_obs, "robot_2/puck_y_pos")[:] = puck_pos_2[1]
        self.obs_helper.get_from_obs(new_obs, "robot_2/puck_yaw_pos")[:] = puck_pos_2[2]

        self.obs_helper.get_from_obs(new_obs, "robot_2/puck_x_vel")[:] = puck_vel_2[0]
        self.obs_helper.get_from_obs(new_obs, "robot_2/puck_y_vel")[:] = puck_vel_2[1]
        self.obs_helper.get_from_obs(new_obs, "robot_2/puck_yaw_vel")[:] = puck_vel_2[2]

        opponent_pos_2 = self.obs_helper.get_from_obs(new_obs, 'robot_2/opponent_ee_pos')
        self.obs_helper.get_from_obs(new_obs, 'robot_2/opponent_ee_pos')[:] = \
            (np.linalg.inv(self.env_info['robot']['base_frame'][1]) @ np.concatenate([opponent_pos_2, [1]]))[:3]

        return new_obs

    def setup(self, obs):
        for i in range(7):
            self._data.joint("iiwa_1/joint_" + str(i + 1)).qpos = self.init_state[i]
            self._data.joint("iiwa_2/joint_" + str(i + 1)).qpos = self.init_state[i]

            self.q_pos_prev[i] = self.init_state[i]
            self.q_pos_prev[i + 7] = self.init_state[i]
            self.q_vel_prev[i] = self._data.joint("iiwa_1/joint_" + str(i + 1)).qvel
            self.q_vel_prev[i + 7] = self._data.joint("iiwa_2/joint_" + str(i + 1)).qvel

        self.universal_joint_plugin.reset()

        super().setup(obs)
        # Update body positions, needed for _compute_universal_joint
        mujoco.mj_fwdPosition(self._model, self._data)

    def reward(self, state, action, next_state, absorbing):
        return 0


def main():
    env = AirHockeyDouble(viewer_params={'start_paused': True})
    env.reset()
    env.render()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.zeros(14)
        observation, reward, done, info = env.step(action)
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()


if __name__ == '__main__':
    main()
