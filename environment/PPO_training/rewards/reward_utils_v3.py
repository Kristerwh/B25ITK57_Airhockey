import numpy as np

def proximity_to_puck(puck_pos, mallet_pos):
    dist = np.linalg.norm(puck_pos - mallet_pos)
    return max(0, 1.0 - dist) * 0.5 if puck_pos[0] < 0 else 0

def move_toward_puck(mallet_vel, puck_pos, mallet_pos):
    direction = puck_pos - mallet_pos
    if np.linalg.norm(direction) == 0:
        return 0
    direction /= (np.linalg.norm(direction) + 1e-8)
    alignment = np.dot(mallet_vel, direction)
    return 0.3 * alignment

def positioning_penalty(mallet_pos):
    if mallet_pos[0] > 0:
        return -4.0 - 8.0 * mallet_pos[0]
    elif mallet_pos[0] > -0.1:
        return -0.2
    return 0

def hit_puck(distance):
    return 0.9 if distance < 0.1 else 0

def small_penalties(distance, mallet_vel):
    return -0.05 * distance + 0.01 * np.linalg.norm(mallet_vel)

def behind_puck(puck_pos, mallet_pos):
    return 0.2 if puck_pos[0] < mallet_pos[0] and mallet_pos[0] < 0 else 0

def push_direction_bonus(puck_push):
    return 4.0 * puck_push if puck_push > 0 else 0

def push_direction_penalty(puck_push):
    if puck_push < 0:
        return -6.0 * abs(puck_push) - 10.0 * abs(puck_push) ** 2
    return 0

def puck_velocity_bonus(puck_vel):
    return 0.2 * np.linalg.norm(puck_vel)

def puck_alignment(puck_vel):
    norm = np.linalg.norm(puck_vel)
    if norm == 0:
        return 0
    return 0.5 * np.dot(puck_vel / norm, np.array([1.0, 0.0]))

def urgency_penalty(obs, next_obs, puck_vel, mallet_pos, puck_pos):
    if obs[0] > -0.4 and next_obs[0] < -0.6 and puck_vel[0] < 0 and np.linalg.norm(mallet_pos - puck_pos) > 0.3:
        return -0.5
    return 0

def clearance_reward(obs, next_obs, puck_vel):
    if obs[0] < -0.6 and next_obs[0] > -0.3 and puck_vel[0] > 0:
        return 1.5
    return 0

def goal_event_score(next_obs, absorbing):
    if next_obs[0] > 0 and absorbing:
        return 40
    return 0

def goal_event_conceded(next_obs, absorbing):
    if next_obs[0] <= 0 and absorbing:
        return -60
    return 0

def puck_to_goal_velocity(puck_vel, next_obs):
    if puck_vel[0] > 0 and next_obs[0] > 0:
        return 0.5 * puck_vel[0]
    return 0

def near_goal_bonus(next_obs):
    if next_obs[0] > 0.85 and abs(next_obs[1]) < 0.2:
        return 2.0
    return 0

def stuck_puck_penalty(next_obs, puck_vel):
    if next_obs[0] < -0.6 and np.linalg.norm(puck_vel) < 0.01:
        return -1.0
    return 0

def velocity_alignment_bonus(puck_pos, mallet_pos, mallet_vel):
    direction = puck_pos - mallet_pos
    if np.linalg.norm(direction) == 0:
        return 0
    direction /= (np.linalg.norm(direction) + 1e-8)
    return 0.1 * np.dot(mallet_vel, direction)

def wall_penalty(next_obs):
    return -1.0 if abs(next_obs[1]) > 0.9 else 0

def direction_change_penalty(obs, next_obs):
    if np.sign(obs[2]) != np.sign(next_obs[2]) and abs(obs[2]) > 0.1 and abs(next_obs[2]) > 0.1:
        return -2.0
    return 0

def passive_goal_defense(puck_vel, next_obs, mallet_vel):
    if puck_vel[0] < -0.1 and next_obs[0] < -0.6 and np.linalg.norm(mallet_vel) < 0.05:
        return -1.0
    return 0

def puck_trap_bonus(next_obs, puck_pos, mallet_pos):
    if next_obs[0] < -0.7 and np.linalg.norm(puck_pos - mallet_pos) < 0.15:
        return 1.5
    return 0

def minimal_motion_penalty(puck_vel, mallet_vel):
    if np.linalg.norm(puck_vel) < 0.01 and np.linalg.norm(mallet_vel) < 0.01:
        return -2.0
    return 0

def small_energy_penalty():
    return -0.02