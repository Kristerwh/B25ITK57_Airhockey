
import numpy as np

def proximity_to_puck(obs, action, next_obs, done, env, ep):
    puck = obs[:2]
    mallet = obs[4:6]
    distance = np.linalg.norm(puck - mallet)
    on_own_half = puck[0] < 0
    if on_own_half:
        return max(0, 0.9 * (1 - distance / 0.3))  # cap around 0.3m
    return 0

def move_toward_puck(obs, action, next_obs, done, env, ep):
    puck = obs[:2]
    mallet = obs[4:6]
    mallet_vel = obs[6:8]
    direction = puck - mallet
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 0
    unit_direction = direction / norm
    speed_toward = np.dot(mallet_vel, unit_direction)
    return np.clip(speed_toward / 2.0, 0.1, 0.3) if speed_toward > 0 else 0

def hit_puck(obs, action, next_obs, done, env, ep):
    puck = obs[:2]
    mallet = obs[4:6]
    distance = np.linalg.norm(puck - mallet)
    if distance < 0.1:
        return 0.9
    return 0

def push_right(obs, action, next_obs, done, env, ep):
    puck_vel = next_obs[2:4]
    if puck_vel[0] > 0:
        return min(4.0, puck_vel[0] * 10)
    return 0

def push_wrong(obs, action, next_obs, done, env, ep):
    puck_vel = next_obs[2:4]
    if puck_vel[0] < 0:
        return max(-10, puck_vel[0] * 20)
    return 0

def puck_speed(obs, action, next_obs, done, env, ep):
    speed = np.linalg.norm(next_obs[2:4])
    return np.clip(speed, 0, 0.5)

def puck_direction(obs, action, next_obs, done, env, ep):
    puck_vel = next_obs[2:4]
    goal_dir = np.array([1.0, 0.0])
    dot = np.dot(puck_vel, goal_dir)
    if np.linalg.norm(puck_vel) == 0:
        return 0
    return max(0, dot / np.linalg.norm(puck_vel)) * 0.5

def passive_penalty(obs, action, next_obs, done, env, ep):
    puck = obs[:2]
    mallet = obs[4:6]
    distance = np.linalg.norm(puck - mallet)
    mallet_vel = obs[6:8]
    if distance < 0.3 and np.linalg.norm(mallet_vel) < 0.1:
        return -0.5
    return 0

def behind_puck(obs, action, next_obs, done, env, ep):
    puck_x = obs[0]
    mallet_x = obs[4]
    if mallet_x < puck_x:
        return 0.2
    return 0

def overposition(obs, action, next_obs, done, env, ep):
    mallet_x = obs[4]
    if mallet_x > 0:
        return -min(5.0, mallet_x * 10)
    return 0

def goal_scored(obs, action, next_obs, done, env, ep):
    puck_x = next_obs[0]
    if puck_x > 0 and done:
        return 40
    return 0

def goal_conceded(obs, action, next_obs, done, env, ep):
    puck_x = next_obs[0]
    if puck_x < 0 and done:
        return -60
    return 0
