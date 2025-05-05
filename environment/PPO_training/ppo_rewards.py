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

    #Mallet position constraint, if agent moves outside its side of the table = bad behaviour
    if mallet_pos[0] > 0.3:
        reward -= 5.0

    return reward

def phase1_reward(obs, action, next_obs, absorbing, env, ep=None):
    puck_pos = obs[0:2]
    puck_vel = obs[2:4]
    mallet_pos = obs[4:6]
    mallet_vel = obs[6:8]

    reward = shaping_rewards(obs, mallet_pos, mallet_vel, puck_pos, puck_vel)
    #time penalty, stay still = bad behaviour
    reward -= 0.01
    return np.clip(reward, -10, 10)

def phase2_reward(obs, action, next_obs, absorbing, env, ep=None):
    reward = phase1_reward(obs, action, next_obs, absorbing, env, ep)
    puck_push = next_obs[0] - obs[0]
    reward += 3.0 * puck_push
    reward += 0.2 * np.linalg.norm(obs[2:4])
    return np.clip(reward, -20, 20)

def phase3_reward(obs, action, next_obs, absorbing, env, ep=None):
    reward = phase2_reward(obs, action, next_obs, absorbing, env, ep)

    if hasattr(env, "is_colliding") and env.is_colliding(next_obs, 'puck', 'paddle_left'):
        reward += 15.0

    puck_push = next_obs[0] - obs[0]
    reward += 5.0 * obs[2]

    if next_obs[0] > 0 and absorbing:
        reward += 50
    if next_obs[0] <= 0 and absorbing:
        reward -= 50

    if obs[7] < -0.2:
        reward -= 0.2

    if np.linalg.norm(obs[2:4]) < 0.01:
        reward -= 1.0

    reward -= 0.01
    return np.clip(reward, -50, 50)
