import mujoco
import glfw
import numpy as np


class MuJoCoPhysics:
    def __init__(self, xml_path):
        with open(xml_path, "r") as f:
            self.model = mujoco.MjModel.from_xml_string(f.read())
        self.data = mujoco.MjData(self.model)

        if not glfw.init():
            raise Exception("No GLFW")

        self.window = glfw.create_window(800, 800, "MuJoCo Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window could not be created!")

        glfw.make_context_current(self.window)

        self.renderer = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.camera = mujoco.MjvCamera()


        self.camera.azimuth = 180
        self.camera.elevation = -30
        self.camera.distance = 2.5
        self.camera.lookat = np.array([0, 0, 0.05])

        self.options = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_position(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id][:2].tolist()

    #testing with actuator instead of apply force with xfrc_
    def actuator_control(self, actuator_name, control_value):
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        self.data.ctrl[actuator_id] = control_value

    def stop_actuators(self, actuator_x, actuator_y):
        self.actuator_control(actuator_x, 0)
        self.actuator_control(actuator_y, 0)

    def check_collision(self, body_name1, body_name2):
        body_id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name1)
        body_id2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name2)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == body_id1 and contact.geom2 == body_id2) or \
                    (contact.geom1 == body_id2 and contact.geom2 == body_id1):
                return True
        return False

    def set_paddle_position(self, body_name, position):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]

        position[0] = np.clip(position[0], -1.15, 1.15)
        position[1] = np.clip(position[1], -0.55, 0.55)

        self.data.qpos[joint_id:joint_id + 2] = position
        self.data.qvel[joint_id:joint_id + 2] *= 0.9

    def render(self):
        if glfw.window_should_close(self.window):
            return False

        # glfw.make_context_current(self.window)

        mujoco.mjv_updateScene(self.model, self.data, self.options, self.perturb, self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 800), self.scene, self.renderer)

        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return True

    def close(self):
        glfw.terminate()
