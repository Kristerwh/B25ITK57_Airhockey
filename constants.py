#constants that we can reuse on different files/versions of the simulation later. We can add and remove as we wish
#We should add future constants here as well, it will keep the project clean, and we can easily change parameters later
WIDTH = 1200
HEIGHT = 800

COLORS = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0),
    "green": (0, 255, 0)
}


#paddles and puck
paddle_radius = 30
paddle_speed = 50

paddle1_start_location = [-0.4, 0]
paddle2_start_location = [0.4, 0]
paddle_move_range = 0.0014

puck_radius = 20
puck_start_location = [0, 0, 0.02]
puck_speed = [0, 0, 0]

#Physics settings for later, these are not the actual physics data
FRICTION = 0.98
ELASTICITY = 0.9
