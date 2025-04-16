import numpy as np

def ppo_reward(obs, action, next_obs, absorbing, env, ep=None):
    puck_pos = obs[0:2]
    puck_vel = obs[2:4]
    mallet_pos = obs[4:6]
    mallet_vel = obs[6:8]

    next_puck_pos = next_obs[0:2]
    puck_speed = np.linalg.norm(puck_vel)
    puck_push = next_puck_pos[0] - puck_pos[0]

    reward = 0
    hit = False

    #---------------- INIT counters if needed ----------------
    if not hasattr(env, "idle_puck_counter"):
        env.idle_puck_counter = 0
    if not hasattr(env, "no_hit_counter"):
        env.no_hit_counter = 0

    #---------------- PHASE CONTROL ----------------
    if ep is None:
        ep = 0

    shaping_active = ep < 300
    fade = 1.0 if ep < 200 else max(0.0, 1 - (ep - 200) / 100)

    #---------------- SHAPING REWARDS (FADING) ----------------

    #1. Distance shaping
    if shaping_active:
        distance = np.linalg.norm(puck_pos - mallet_pos)
        reward += fade * max(0, 1.0 - distance) * 0.2

    #2. Velocity shaping
    direction_to_puck = puck_pos - mallet_pos
    velocity_dir = 0.0
    if np.linalg.norm(direction_to_puck) > 0:
        direction_to_puck /= (np.linalg.norm(direction_to_puck) + 1e-8)
        velocity_dir = np.dot(mallet_vel, direction_to_puck)
        reward += fade * 0.3 * velocity_dir

    #3. Stay on left side
    if shaping_active:
        if mallet_pos[0] < puck_pos[0]:
            reward += fade * 0.5
        if mallet_pos[0] > 0:
            reward -= fade * 0.5

    #4. Avoid hugging bottom wall
    if mallet_pos[1] < -0.85:
        reward -= 1.0

    #---------------- DYNAMIC HIT REWARD ----------------

    if hasattr(env, "is_colliding") and env.is_colliding(next_obs, 'puck', 'paddle_left'):
        reward += 15.0
        hit = True

    #---------------- PUCK MOVEMENT / SPEED ----------------

    if puck_push <= 0.001:
        reward -= 2.0
    else:
        reward += 3.0 * puck_push

    reward += 0.2 * puck_speed
    reward += 5.0 * puck_vel[0]  # toward opponent goal

    #---------------- GOAL REWARDS ----------------

    if next_puck_pos[0] > 0 and absorbing:
        reward += 50
    if next_puck_pos[0] <= 0 and absorbing:
        reward -= 50

    #---------------- PENALTIES ----------------

    if mallet_vel[1] < -0.2:
        reward -= 0.2

    reward -= 0.01  # time penalty

    #Anti-idle: puck not moving
    if np.linalg.norm(puck_vel) < 0.01:
        env.idle_puck_counter += 1
    else:
        env.idle_puck_counter = 0

    if env.idle_puck_counter > 20:
        reward -= 1.0

    #Anti-idle: not hitting
    if not hit:
        env.no_hit_counter += 1
    else:
        env.no_hit_counter = 0

    if env.no_hit_counter > 100:
        reward -= 1.0

    #---------------- HARD CLAMP ----------------
    if not hit and puck_push < 0.01:
        reward = min(reward, -2.0)

    reward = np.clip(reward, -50, 50)

    #Debug
    if np.random.rand() < 0.01:
        print(f"[REWARD DEBUG] ep={ep} reward={reward:.2f} fade={fade:.2f} hit={hit} push={puck_push:.2f} vel_y={mallet_vel[1]:.2f}")

    return reward
