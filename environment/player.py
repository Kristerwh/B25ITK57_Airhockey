import mujoco
import numpy as np
import glfw


class Player:
    def __init__(self, physics, paddle_name):
        self.physics = physics
        self.paddle_name = paddle_name

    def move(self):
        position = self.physics.get_position(self.paddle_name)
        self.physics.set_position(self.paddle_name, position)