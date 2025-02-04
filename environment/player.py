import mujoco
import numpy as np


class Player:
    def __init__(self, physics, body_name,actuator_x, actuator_y,ai_player=False, mouse_control=False):
        self.physics = physics
        self.body_name = body_name
        self.ai_player = ai_player
        self.velocity = [0, 0]
        self.actuator_x = actuator_x
        self.actuator_y = actuator_y
        self.mouse_control = mouse_control
        self.is_grabbed = False


    def controls(self, direction):
        acceleration = 0.1

        if direction == "down":
            self.physics.actuator_control(self.actuator_y, acceleration)
        elif direction == "up":
            self.physics.actuator_control(self.actuator_y, -acceleration)
        elif direction == "left":
            self.physics.actuator_control(self.actuator_x, -acceleration)
        elif direction == "right":
            self.physics.actuator_control(self.actuator_x, acceleration)
        else:
            self.stop()

    def stop(self):
        self.physics.actuator_control(self.actuator_x, 0)
        self.physics.actuator_control(self.actuator_y, 0)

    def mouse_control_movement(self, x, y, mouse_pressed):
        if self.mouse_control:
            if mouse_pressed:
                self.is_grabbed = True
            elif not mouse_pressed:
                self.is_grabbed = False

            if self.is_grabbed:
                if np.isnan(x) or np.isnan(y):
                    print("error")
                    return
                if np.isinf(x) or np.isinf(y):
                    print("error")
                    return

                clamped_x = max(-1.15, min(1.15, x))
                clamped_y = max(-0.55, min(0.55, y))
                self.physics.set_paddle_position(self.body_name, [clamped_x, clamped_y])
