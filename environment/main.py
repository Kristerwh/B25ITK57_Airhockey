import glfw
from constraints import Constraints
from position_control_wrapper import PositionControl
from challenge_core import ChallengeCore
import mujoco
import mujoco.viewer
from environment.physics_mujoco import MuJoCoPhysics
from environment.player import Player


class PlaceholderMDP:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("airhockey_table.xml")
        self.data = mujoco.MjData(self.model)
        self.info = self
        self.dt = 0.02

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return None

    def step(self, action):
        mujoco.mj_step(self.model, self.data)
        return None, 0, False, {}

    def render(self, viewer):
        viewer.sync()

mouse_x, mouse_y = 0, 0

def move_paddle_with_mouse(model, data, action, reldx, reldy, scene, perturb):
    mujoco.mjv_movePerturb(model, data, action, reldx, reldy, scene, perturb)

def main():
    physics = MuJoCoPhysics("airhockey_table.xml")
    viewer = mujoco.viewer.launch_passive(physics.physics, physics.data)

    scene = mujoco.MjvScene(physics.physics, maxgeom=1000)
    perturb = mujoco.MjvPerturb()
    mujoco.mjv_defaultPerturb(perturb)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    mujoco.mjv_updateScene(physics.physics, physics.data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL,scene)

    mdp = PlaceholderMDP()

    player1 = Player(physics, "paddle1")
    player2 = Player(physics, "paddle2")

    game = ChallengeCore(agent=None, mdp=mdp)

    while True:
        while True:
            move_paddle_with_mouse(physics.physics, physics.data, mujoco.mjtMouse.mjMOUSE_MOVE_H, 0.01, 0.01, scene,perturb)
            mujoco.mjv_applyPerturbPose(physics.physics, physics.data, perturb, 1)
            mujoco.mj_step(physics.physics, physics.data)
            mujoco.mjv_updateScene(physics.physics, physics.data, mujoco.MjvOption(), None, camera,mujoco.mjtCatBit.mjCAT_ALL, scene)
            viewer.sync()

            #puck collision
            puck_pos = physics.get_position("puck")
            if puck_pos is not None:
                puck_pos = Constraints.enforce_table_bounds(puck_pos)
                physics.set_position("puck", puck_pos)

            #Apply physics constraints (friction, bounce, speed limit)
            puck_velocity = physics.get_velocity("puck")
            puck_velocity = Constraints.apply_physics_constraints(puck_velocity)
            physics.set_velocity("puck", puck_velocity)

            #paddle range, cant go beyond middle
            paddle1_pos = physics.get_position("paddle1")
            paddle1_pos = Constraints.enforce_paddle_bounds(paddle1_pos, "left")
            physics.set_position("paddle1", paddle1_pos)

            paddle2_pos = physics.get_position("paddle2")
            paddle2_pos = Constraints.enforce_paddle_bounds(paddle2_pos, "right")
            physics.set_position("paddle2", paddle2_pos)

            mujoco.mjv_applyPerturbPose(physics.physics, physics.data, perturb, 1)
            mujoco.mj_step(physics.physics, physics.data)
            viewer.sync()

if __name__ == "__main__":
    main()