import math


S1=10
S2 = 10
T = 1

DBOX_Y = 200,400
DBOX_X = 200



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
    reward_amount = 0
    if len(puck_array) - collision_idx > 2 and collision_idx  > 2:
        x1, y1 = puck_array[0]
        x2, y2 = puck_array[collision_idx]
        x3, y3 = puck_array[-1]

        vector_before_hit = (x2-x1), (y2-y1)
        vector_after_hit = (x3-x2), (y3-y2)

        vx2,_ = vector_after_hit
        if vx2 > 0:
            print("good hit")
            reward_amount =+ 1
        if vx2 < 0:
            print("bad hit")
            reward_amount =- 1
    return reward_amount



def collision_reward(puck_pos_array, mallet_pos_array):
    reward_amount = 0
    collision_idx = detect_collision_two_circles(puck_pos_array,mallet_pos_array,S1,S2,T)
    if collision_idx is not None:
        reward_amount += 1
        reward_amount =+ get_directions(collision_idx,puck_pos_array)
    return reward_amount


def save_reward(collision_idx,puck_pos_array):
    reward_amount = 0
    x,y = puck_pos_array[collision_idx]
    if x < DBOX_X and DBOX_Y[0] < y < DBOX_Y[1]:
        reward_amount += 1
        print("good save!")
    return reward_amount

def shot_at_goal(puck_pos_array):
    reward_amount = 0
    idx = 0
    while idx < (len(puck_pos_array) - 1):
        x1, y1 = puck_pos_array[idx]
        x2, y2 = puck_pos_array[idx+1]

        vx = x2 - x1
        vy = y2 - y1

        if vx > 0:
            while x2 > 0:
                x2 += vx
                y2 += vy
            if DBOX_Y[0] < y2 < DBOX_Y[1]:
                reward_amount =+ 1
                print("shot at goal!")
    return reward_amount



puck_pos = 100,100
mallet_pos = 50,50
mallet_vel = 1,1

def reward_for_moving_toward_puck():
    px, py = puck_pos
    mx, my = mallet_pos
    mvx, mvy = mallet_vel

    # Compute direction difference
    dx, dy = px - mx, py - my

    # Check if mallet is moving in the correct direction
    if (dx * mvx >= 0) and (dy * mvy >= 0):
        return 1  # Reward for moving toward the puck
    else:
        return -1  # Penalty for moving away


