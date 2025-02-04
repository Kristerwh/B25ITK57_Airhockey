# import mujoco
# import numpy as np
#
# from constants import puck_radius, paddle_radius
#
#
# class Puck:
#     def __init__(self, physics, object_name="puck"):
#         self.physics = physics
#         self.object_name = object_name
#
#     def get_position(self):
#         return self.physics.get_position(self.object_name)
#
#     def reset_position(self):
#         self.physics.set_position(self.object_name, [0, 0, 0])