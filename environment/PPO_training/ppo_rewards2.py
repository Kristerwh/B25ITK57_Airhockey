import numpy as np

def shaping_rewards(obs, mallet_pos, mallet_vel, puck_pos, puck_vel, fade=1.0):
    reward = 0
    distance = np.linalg.norm(puck_pos - mallet_pos)
    reward += fade * max(0, 1.0 - distance) * 0.2

    direction_to_puck = puck_pos - mallet_pos
    if np.linalg.norm(direction_to_puck) > 0:
        direction_to_puck /= (np.linalg.norm(direction_to_puck) + 1e-8)
        velocity_dir = np.dot(mallet_vel, direction_to_puck)
        reward += fade * 0.3 * velocity_dir

    if mallet_pos[0] < puck_pos[0]:
        reward += fade * 0.5
    if mallet_pos[0] > 0:
        reward -= fade * 0.5

    if mallet_pos[1] < -0.85:
        reward -= 1.0

    if mallet_pos[0] > 0.3:
        reward -= 5.0

    #Prevent sitting on puck
    distance = np.linalg.norm(puck_pos - mallet_pos)
    if distance < 0.01:
        reward -= 5.0

    return reward

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
            alignment = np.dot(mallet_vel, direction_to_puck)
            reward += 0.3 * alignment
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

    if puck_push > 0:
        reward += 4.0 * puck_push
    else:
        penalty = 6.0 * abs(puck_push) + 10.0 * abs(puck_push) ** 2
        reward -= penalty

    reward += 0.2 * np.linalg.norm(puck_vel)

    vel_dir = puck_vel / (np.linalg.norm(puck_vel) + 1e-8)
    alignment = np.dot(vel_dir, np.array([1.0, 0.0]))
    reward += 0.5 * alignment

    #Mild urgency — penalize passivity when puck is moving toward agent
    if obs[0] > -0.4 and next_obs[0] < -0.6 and puck_vel[0] < 0 and np.linalg.norm(mallet_pos - puck_pos) > 0.3:
        reward -= 0.5  # didn't act to intercept

    return np.clip(reward, -20, 20)

def phase3_reward(obs, action, next_obs, absorbing, env, ep=None):
    reward = phase2_reward(obs, action, next_obs, absorbing, env, ep)

    puck_pos = next_obs[0:2]
    puck_vel = next_obs[2:4]
    mallet_pos = next_obs[4:6]
    mallet_vel = next_obs[6:8]

    if next_obs[0] < 0:
        distance = np.linalg.norm(puck_pos - mallet_pos)
        if distance < 0.6:
            reward += max(0, 1.0 - distance) * 0.2

    if puck_vel[0] > 0 and next_obs[0] > 0:
        reward += 0.5 * puck_vel[0]

    if next_obs[0] > 0.85 and abs(next_obs[1]) < 0.2:
        reward += 2.0

    if next_obs[0] < -0.6 and np.linalg.norm(puck_vel) < 0.01:
        reward -= 1.0

    direction = puck_pos - mallet_pos
    if np.linalg.norm(direction) > 0:
        direction /= (np.linalg.norm(direction) + 1e-8)
        alignment = np.dot(mallet_vel, direction)
        reward += 0.1 * alignment

    if next_obs[0] > 0 and absorbing:
        reward += 40
    elif next_obs[0] <= 0 and absorbing:
        reward -= 60

    if next_obs[0] < -0.5:
        reward -= 0.2

    delta_x = next_obs[0] - obs[0]
    if delta_x > 0:
        reward += delta_x * 2.0
    else:
        reward -= abs(delta_x) * 0.5

    if next_obs[0] < -0.7 and np.linalg.norm(puck_pos - mallet_pos) < 0.15:
        reward += 1.5

    if np.linalg.norm(puck_vel) < 0.01 and np.linalg.norm(mallet_vel) < 0.01:
        reward -= 2.0

    reward += 0.2 * np.linalg.norm(puck_vel)

    reward -= 0.02

    if abs(next_obs[1]) > 0.9:
        reward -= 1.0

    if np.sign(obs[2]) != np.sign(next_obs[2]) and abs(obs[2]) > 0.1 and abs(next_obs[2]) > 0.1:
        reward -= 2.0

    #Penalize passivity while puck is moving toward own goal
    if puck_vel[0] < -0.1 and next_obs[0] < -0.6 and np.linalg.norm(mallet_vel) < 0.05:
        reward -= 1.0

    #Reward successful clearance
    if obs[0] < -0.6 and next_obs[0] > -0.3 and puck_vel[0] > 0:
        reward += 1.5

    return np.clip(reward, -100, 100)
