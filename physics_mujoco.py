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


        # self.camera.azimuth = 360
        # self.camera.elevation = -50  # Less tilt to center the view
        # self.camera.distance = 1.5  # Zoom out slightly
        # self.camera.lookat = np.array([0.05, 0.05, 0.05])

        self.options = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_position(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id][:2].tolist()

    def apply_force(self, body_name, force):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.data.xfrc_applied[body_id][:2] = force

    def render(self):
        if glfw.window_should_close(self.window):
            return False

        # glfw.make_context_current(self.window)

        mujoco.mjv_updateScene(self.model, self.data, self.options, self.perturb, self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 100, 800, 800), self.scene, self.renderer)

        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return True

    def close(self):
        glfw.terminate()
