import numpy as np
import math

MALLET_SPEED = 100
TABLE_WIDTH = 1948
TABLE_HEIGHT = 1038
PUCK_POSITION_ARRAY_SIZE = 10
MALLET_POSITION_ARRAY_SIZE = 10
MALLET_HOME_POS = 100 ,(TABLE_HEIGHT / 2)


def calculate_distance(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


class AirHockeyAIv4:
    def __init__(self):
        self.mallet_positions = []
        self.mallet_current_pos = None
        self.puck_positions = []
        self.puck_current_pos = None

    def update_puck_positions (self, new_puck_position):
        self.puck_current_pos = new_puck_position
        if len(self.puck_positions) >= PUCK_POSITION_ARRAY_SIZE:
            self.puck_positions.pop(0)
        self.puck_positions.append(new_puck_position)
        if len (self.puck_positions) <= 1:
            return True

    def update_mallet_positions (self, new_mallet_position):
        self.mallet_current_pos = new_mallet_position
        if len(self.mallet_positions) >= MALLET_POSITION_ARRAY_SIZE:
            self.mallet_positions.pop(0)
        self.mallet_positions.append(new_mallet_position)

    def move_mallet_home(self):
        vector = self.mallet_current_pos - MALLET_HOME_POS
        distance = calculate_distance(vector)
        scale = distance / MALLET_SPEED
        vx = vector[0] * scale
        vy = vector[1] * scale
        return vx, vy, scale

    def calculate_puck_trajectory(self):
        vector = self.puck_positions[0] - self.puck_positions[-1]
        predicted_pos = vector + self.puck_current_pos
        distance = calculate_distance(predicted_pos)
        if predicted_pos[0] < 0:
            predicted_pos[0] = -predicted_pos[0]
        if predicted_pos[0] > TABLE_WIDTH:
            tmp = predicted_pos[0] - TABLE_WIDTH
            predicted_pos[0] = predicted_pos[0] - 2 * tmp
        if predicted_pos[1] < 0:
            predicted_pos[1] = -predicted_pos[1]
        if predicted_pos[1] > TABLE_HEIGHT:
            tmp = predicted_pos[1] - TABLE_HEIGHT
            predicted_pos[1] = predicted_pos[1] - 2 * tmp
        return predicted_pos, distance

    def general_action(self):
        pos, dist = self.calculate_puck_trajectory()
        if pos[0] < TABLE_WIDTH:
            pass







def startup():
    ai = AirHockeyAIv4()
    return ai

#will only return vx, vy for mallet
def run(ai, puck_pos, mallet_pos):

    ai.update_mallet_positions(mallet_pos)
    if ai.update_puck_positions(puck_pos): #returns true if the array size less than 2
        return 0,0

