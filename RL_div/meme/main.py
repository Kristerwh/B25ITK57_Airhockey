import math

S1=10
S2 = 10
T = 1

def detect_collision_two_circles(pos_circle_1, pos_circle_2, size_circle_1, size_circle_2, tolerance):
    distance = size_circle_1 + size_circle_2 + tolerance
    if len(pos_circle_1) != len(pos_circle_2):
        raise ValueError("Array size uneven!")
    idx = 0
    while idx < len(pos_circle_1):
        x1, y1 = pos_circle_1[idx]
        x2, y2 = pos_circle_2[idx]
        x = abs (x1 - x2)
        y = abs (y1 - y2)
        hypot = math.hypot(x,y)
        if hypot < distance:
            return idx
        idx += 1
    return None

def get_directions(collision_idx,puck_array):
    if len(puck_array) - collision_idx > 2 and collision_idx  > 2:
        x1, y1 = puck_array[0]
        x2, y2 = puck_array[collision_idx]
        x3, y3 = puck_array[-1]

        vector_before_hit = (x2-x1), (y2-y1)
        vector_after_hit = (x3-x2), (y3-y2)

        vx1,_ = vector_before_hit
        vx2,_ = vector_after_hit
        if vx1 < 0 and vx2 < 0:
            print("good hit")
            return True
    return False


def collision_reward(puck_pos_array, mallet_pos_array):
    reward_amount = 0
    collision_idx = detect_collision_two_circles(puck_pos_array,mallet_pos_array,S1,S2,T)
    if collision_idx is not None:
        reward_amount += 1


    pass