import numpy as np

def phase1_reward(obs, action, next_obs, absorbing, env):
    puck_pos = obs[0:2]
    mallet_pos = obs[4:6]
    mallet_vel = obs[6:8]
    next_puck_pos = next_obs[0:2]

    distance = np.linalg.norm(puck_pos - mallet_pos)
    push = next_puck_pos[0] - puck_pos[0]

    reward = 0
    #approach puck
    reward += max(0, 1.0 - distance) * 1.0
    if distance < 0.15:
        #touch
        reward += 5.0
    if push > 0:
        #push puck forward
        reward += 3.0 * push
    #time penalty
    reward -= 0.01

    if np.random.rand() < 0.01:
        print(f"[PHASE1 DEBUG] reward={reward:.2f} dist={distance:.2f} push={push:.2f}")

    return reward

def ppo_reward(obs, action, next_obs, absorbing, env):
    puck_pos = obs[0:2]
    puck_vel = obs[2:4]
    mallet_pos = obs[4:6]
    mallet_vel = obs[6:8]

    next_puck_pos = next_obs[0:2]
    puck_speed = np.linalg.norm(puck_vel)
    puck_push = next_puck_pos[0] - puck_pos[0]

    reward = 0

    #Reduced proximity reward
    distance = np.linalg.norm(puck_pos - mallet_pos)
    reward += max(0, 1.0 - distance) * 0.2

    #Move toward puck
    direction_to_puck = puck_pos - mallet_pos
    velocity_dir = 0.0
    if np.linalg.norm(direction_to_puck) > 0:
        direction_to_puck /= (np.linalg.norm(direction_to_puck) + 1e-8)
        velocity_dir = np.dot(mallet_vel, direction_to_puck)
        reward += 0.3 * velocity_dir

    #Defensive side
    if mallet_pos[0] < puck_pos[0]:
        reward += 0.5
    if mallet_pos[0] > 0:
        reward -= 0.5

    hit = False
    if hasattr(env, "is_colliding") and env.is_colliding(next_obs, 'puck', 'paddle_left'):
        reward += 15.0
        hit = True

    #Puck push or penalize
    if puck_push <= 0.001:
        reward -= 5.0
    else:
        reward += 3.0 * puck_push

    reward += 0.2 * puck_speed

    if next_puck_pos[0] > 0 and absorbing:
        reward += 50
    if next_puck_pos[0] <= 0 and absorbing:
        reward -= 50

    if mallet_vel[1] < -0.1:
        reward -= 0.1

    reward -= 0.01

    if np.random.rand() < 0.01:
        print(f"[REWARD DEBUG] total={reward:.2f} dist={distance:.2f} vel_dot={velocity_dir:.2f} mallet_x={mallet_pos[0]:.2f} vel_y={mallet_vel[1]:.2f} push={puck_push:.2f} puck_speed={puck_speed:.2f} hit={hit}")

    return reward