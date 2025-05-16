import numpy as np

def phase1_reward(obs, action, next_obs, absorbing, env, ep=None):
    puck_pos = obs[0:2]
    puck_vel = obs[2:4]
    mallet_pos = obs[4:6]
    mallet_vel = obs[6:8]
    reward = 0
    distance = np.linalg.norm(puck_pos - mallet_pos)

    if puck_pos[0] < 0:
        reward += max(0, 1.0 - distance) * 0.5
        direction_to_puck = puck_pos - mallet_pos
        if np.linalg.norm(direction_to_puck) > 0:
            direction_to_puck /= (np.linalg.norm(direction_to_puck) + 1e-8)
            reward += 0.3 * np.dot(mallet_vel, direction_to_puck)
    else:
        reward -= 0.1 * (mallet_pos[0] > 0)

    if mallet_pos[0] > 0:
        reward -= 4.0 + 8.0 * mallet_pos[0]
    elif mallet_pos[0] > -0.1:
        reward -= 0.2

    if distance < 0.1:
        reward += 0.9

    reward -= 0.05 * distance
    reward += 0.01 * np.linalg.norm(mallet_vel)

    if puck_pos[0] < mallet_pos[0] and mallet_pos[0] < 0:
        reward += 0.2

    return np.clip(reward, -5, 5)

def phase2_reward(obs, action, next_obs, absorbing, env, ep=None):
    reward = phase1_reward(obs, action, next_obs, absorbing, env, ep)

    puck_pos = obs[0:2]
    mallet_pos = obs[4:6]
    puck_push = next_obs[0] - obs[0]
    puck_vel = next_obs[2:4]
    distance = np.linalg.norm(puck_pos - mallet_pos)

    if np.linalg.norm(puck_vel) < 0.01 and distance < 0.4:
        reward -= 2.0

    if puck_push > 0:
        reward += 6.0 * puck_push
    else:
        penalty = 6.0 * abs(puck_push) + 10.0 * abs(puck_push) ** 2
        reward -= penalty

    reward += 0.2 * np.linalg.norm(puck_vel)

    mallet_x, mallet_y = mallet_pos
    if -0.5 < mallet_x < 0.5 and mallet_y < 0:
        reward += 1.5

    vel_dir = puck_vel / (np.linalg.norm(puck_vel) + 1e-8)
    reward += 0.5 * np.dot(vel_dir, np.array([1.0, 0.0]))

    if obs[0] > -0.4 and next_obs[0] < -0.6 and puck_vel[0] < 0 and np.linalg.norm(mallet_pos - puck_pos) > 0.3:
        reward -= 0.5

    return np.clip(reward, -20, 20)

def phase3_reward(obs, action, next_obs, absorbing, env, ep=None):
    reward = 0
    puck_pos = next_obs[0:2]
    puck_vel = next_obs[2:4]
    mallet_pos = next_obs[4:6]
    mallet_vel = next_obs[6:8]
    distance = np.linalg.norm(puck_pos - mallet_pos)

    # Chase & contact
    if next_obs[0] < 0:
        reward += max(0, 1.0 - distance) * 0.5
    if distance < 0.15:
        reward += 1.0
    if distance < 0.01:
        reward -= 3.0

    # Shoot forward
    if puck_vel[0] > 0:
        reward += 3.0 * puck_vel[0]
    if next_obs[0] > 0.85 and abs(next_obs[1]) < 0.2:
        reward += 5.0

    # Concede
    if next_obs[0] < 0 and absorbing:
        reward -= 10.0

    # Clear from danger
    if obs[0] < -0.6 and next_obs[0] > -0.3 and puck_vel[0] > 0:
        reward += 2.0

    # Defense posture
    if -0.5 < mallet_pos[0] < 0.5 and mallet_pos[1] < 0:
        reward += 1.5
    if mallet_pos[0] < -0.8:
        reward -= 1.0

    # Motion incentives
    if np.linalg.norm(mallet_vel) < 0.05 and distance < 0.3:
        reward -= 1.5
    if np.linalg.norm(puck_vel) < 0.01:
        reward -= 1.0
    if np.linalg.norm(mallet_vel) > 0.1:
        reward += 0.2

    # Directional bonus
    direction = puck_pos - mallet_pos
    if np.linalg.norm(direction) > 0:
        direction /= (np.linalg.norm(direction) + 1e-8)
        reward += 0.2 * np.dot(mallet_vel, direction)

    reward -= 0.01  # time penalty

    return np.clip(reward, -50, 100)
