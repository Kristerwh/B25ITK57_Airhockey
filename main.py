import glfw
import pygame
from physics_mujoco import MuJoCoPhysics
from constants import *
from player import *
from puck import *

def main():
    physics = MuJoCoPhysics("air_hockey.xml")
    player1 = Player(physics, "paddle1",paddle1_start_location, ai_player=False)
    player2 = Player(physics, "paddle2",paddle2_start_location, ai_player=False)
    puck = Puck(physics, "puck")

    running = True
    while running:
        physics.step()
        running = physics.render()

        puck_position = puck.get_position()
        player1_position = physics.get_position("paddle1")
        player2_position = physics.get_position("paddle2")

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

        if glfw.get_key(physics.window, glfw.KEY_UP) == glfw.PRESS:
            player2.controls("up")
        elif glfw.get_key(physics.window, glfw.KEY_DOWN) == glfw.PRESS:
            player2.controls("down")
        elif glfw.get_key(physics.window, glfw.KEY_LEFT) == glfw.PRESS:
            player2.controls("left")
        elif glfw.get_key(physics.window, glfw.KEY_RIGHT) == glfw.PRESS:
            player2.controls("right")
        else:
            player2.stop()

        goal = puck.point_scored()
        if goal:
            print(f"{goal}")
            puck.puck_reset()

        puck.handle_wall_collisions()
        puck.apply_friction()

        if puck.check_collision("paddle1"):
            puck.apply_force([0.5, 0.5])
        if puck.check_collision("paddle2"):
            puck.apply_force([-0.5, -0.5])

    physics.close()

if __name__ == "__main__":
    main()