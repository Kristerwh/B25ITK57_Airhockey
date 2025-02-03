import glfw
import pygame
from physics_mujoco import MuJoCoPhysics
from constants import *
from player import *
from puck import *

def main():
    physics = MuJoCoPhysics("air_hockey.xml")

    player1 = Player(physics, "paddle1", "paddle1_actuator_x", "paddle1_actuator_y")
    player2 = Player(physics, "paddle2", "paddle2_actuator_x", "paddle2_actuator_y", mouse_control=True)
    puck = Puck(physics, "puck")

    running = True
    while running:
        physics.step()
        running = physics.render()

        puck_position = puck.get_position()
        player1_position = physics.get_position("paddle1")
        player2_position = physics.get_position("paddle2")

        #red paddle controls
        if glfw.get_key(physics.window, glfw.KEY_W) == glfw.PRESS:
            player1.controls("up")
        elif glfw.get_key(physics.window, glfw.KEY_S) == glfw.PRESS:
            player1.controls("down")
        elif glfw.get_key(physics.window, glfw.KEY_A) == glfw.PRESS:
            player1.controls("left")
        elif glfw.get_key(physics.window, glfw.KEY_D) == glfw.PRESS:
            player1.controls("right")
        else:
            player1.stop()

        # if glfw.get_key(physics.window, glfw.KEY_UP) == glfw.PRESS:
        #     player2.controls("up")
        # elif glfw.get_key(physics.window, glfw.KEY_DOWN) == glfw.PRESS:
        #     player2.controls("down")
        # elif glfw.get_key(physics.window, glfw.KEY_LEFT) == glfw.PRESS:
        #     player2.controls("left")
        # elif glfw.get_key(physics.window, glfw.KEY_RIGHT) == glfw.PRESS:
        #     player2.controls("right")
        # else:
        #     player2.stop()

        #blue paddle controls
        mouse_pressed = glfw.get_mouse_button(physics.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        if mouse_pressed:
            mouse_x, mouse_y = glfw.get_cursor_pos(physics.window)
            win_width, win_height = glfw.get_window_size(physics.window)

            normalized_x = (mouse_y / win_height) * 2.3 - 1.15
            normalized_y = (mouse_x / win_width) * 1.1 - 0.55

            clamped_x = max(-1.15, min(1.15, normalized_x))
            clamped_y = max(-0.55, min(0.55, normalized_y))

            player2.mouse_control_movement(clamped_x, clamped_y, mouse_pressed=True)

        goal = puck.point_scored()
        if goal:
            print(f"{goal}")
            puck.puck_reset()

        puck.handle_wall_collisions()
        puck.apply_friction()

        if puck.check_collision("paddle1"):
            puck.apply_impulse([0.5, 0.5])
        if puck.check_collision("paddle2"):
            puck.apply_impulse([-0.5, -0.5])

    physics.close()

if __name__ == "__main__":
    main()