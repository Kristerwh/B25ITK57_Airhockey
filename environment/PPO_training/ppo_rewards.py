import numpy as np

def phase1_reward(obs, action, next_obs, absorbing, env, ep=None):
    puck_pos = obs[0:2]
    puck_vel = obs[2:4]
    mallet_pos = obs[4:6]
    mallet_vel = obs[6:8]
    reward = 0
    distance = np.linalg.norm(puck_pos - mallet_pos)

    #Proximity reward (puck chasing)
    if puck_pos[0] < 0:
        reward += max(0, 1.5 - distance) * 0.8
        direction_to_puck = puck_pos - mallet_pos
        if np.linalg.norm(direction_to_puck) > 0:
            direction_to_puck /= (np.linalg.norm(direction_to_puck) + 1e-8)
            reward += 0.4 * np.dot(mallet_vel, direction_to_puck)
    else:
        reward -= 0.2 * (mallet_pos[0] > 0)

    #Penalize being too far forward
    if mallet_pos[0] > 0:
        reward -= 5.0 + 10.0 * mallet_pos[0]
    elif mallet_pos[0] > -0.1:
        reward -= 0.4

    #Contact
    if distance < 0.1:
        reward += 1.2

    reward -= 0.07 * distance
    reward += 0.015 * np.linalg.norm(mallet_vel)

    #Defensive zone bonus
    if puck_pos[0] < -0.3 and -0.5 < mallet_pos[0] < 0.5:
        reward += 0.2

    #Idle penalty: if mallet not moving
    if np.linalg.norm(mallet_vel) < 0.03:
        reward -= 0.4
    #Idle & puck is near: even bigger penalty
    if np.linalg.norm(mallet_vel) < 0.02 and distance < 0.4:
        reward -= 0.8

    return np.clip(reward, -7, 7)

def phase2_reward(obs, action, next_obs, absorbing, env, ep=None):
    reward = phase1_reward(obs, action, next_obs, absorbing, env, ep)

    puck_pos = obs[0:2]
    mallet_pos = obs[4:6]
    mallet_vel = obs[6:8]
    puck_push = next_obs[0] - obs[0]
    puck_vel = next_obs[2:4]
    distance = np.linalg.norm(puck_pos - mallet_pos)

    #Encourage agent to hit puck forward
    if np.linalg.norm(puck_vel) < 0.01 and distance < 0.4:
        reward -= 2.5

    if puck_push > 0:
        reward += 8.0 * puck_push
    else:
        penalty = 8.0 * abs(puck_push) + 10.0 * abs(puck_push) ** 2
        reward -= penalty

    reward += 0.25 * np.linalg.norm(puck_vel)

    mallet_x, mallet_y = mallet_pos
    if -0.5 < mallet_x < 0.5 and mallet_y < 0:
        reward += 2.0

    vel_dir = puck_vel / (np.linalg.norm(puck_vel) + 1e-8)
    reward += 0.7 * np.dot(vel_dir, np.array([1.0, 0.0]))

    if obs[0] > -0.4 and next_obs[0] < -0.6 and puck_vel[0] < 0 and np.linalg.norm(mallet_pos - puck_pos) > 0.3:
        reward -= 0.7

    #Penalize agent for standing still too long
    if np.linalg.norm(mallet_vel) < 0.02:
        reward -= 0.5

    return np.clip(reward, -25, 25)

def phase3_reward(obs, action, next_obs, absorbing, env, ep=None):
    reward = 0

    puck_pos = next_obs[0:2]
    puck_vel = next_obs[2:4]
    mallet_pos = next_obs[4:6]
    mallet_vel = next_obs[6:8]
    distance = np.linalg.norm(puck_pos - mallet_pos)
    mallet_speed = np.linalg.norm(mallet_vel)
    puck_speed = np.linalg.norm(puck_vel)

    #Movement & Patrolling
    reward += 0.6 * mallet_speed
    if mallet_speed < 0.01 and distance > 0.3:
        reward -= 1.3
    if mallet_speed < 0.03:
        reward -= 1.7

    #Chase & Seek
    if puck_pos[0] < 0:
        reward += max(0.2, 3.5 - distance) * 1.1
        if distance > 0.5:
            reward -= 5.0
        if mallet_speed < 0.05:
            reward -= 2.8

    #Contact
    if distance < 0.15:
        reward += 4.2
    if distance < 0.01:
        reward -= 6.5
    if distance < 0.05:
        reward += 1.0

    #Offensive shot
    if puck_vel[0] > 0:
        reward += 8.0 * puck_vel[0]
    if next_obs[0] > 0.85 and abs(next_obs[1]) < 0.2:
        reward += 14.0

    #Defensive fail
    if next_obs[0] < 0 and absorbing:
        reward -= 15.0

    #Clear from danger
    if obs[0] < -0.6 and next_obs[0] > -0.3 and puck_vel[0] > 0:
        reward += 5.5

    #Defense zone
    if -0.5 < mallet_pos[0] < 0.5 and -0.4 < mallet_pos[1] < 0.4:
        reward += 3.0

    #Wall penalties
    if abs(mallet_pos[1]) > 0.85:
        reward -= 3.0
    if abs(mallet_pos[1]) > 1.1:
        reward -= 6.0
    if mallet_pos[0] < -0.85:
        reward -= 2.8
    if mallet_pos[0] > 0.2 and puck_pos[0] < 0:
        reward -= 2.6

    #Directional alignment
    direction = puck_pos - mallet_pos
    if np.linalg.norm(direction) > 0:
        direction /= (np.linalg.norm(direction) + 1e-8)
        alignment = np.dot(mallet_vel, direction)
        reward += 0.4 * alignment

    #Wall puck rescue
    if abs(puck_pos[1]) > 0.6 and distance < 0.3:
        reward += 1.3

    #Time penalty (stronger)
    reward -= 0.04

    #Penalty for being far from defense when puck is away
    if puck_pos[0] > 0.3 and mallet_pos[0] < -0.3:
        reward -= 1.0

    #Penalty for being too idle
    if mallet_speed < 0.025:
        reward -= 1.0

    return np.clip(reward, -60, 120)
