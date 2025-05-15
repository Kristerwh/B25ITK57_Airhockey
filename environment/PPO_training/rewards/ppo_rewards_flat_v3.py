from environment.PPO_training.rewards.reward_utils_v3 import *

def flat_reward(obs, action, next_obs, done, env, ep):
    puck_pos = obs[0:2]
    puck_vel = next_obs[2:4]
    mallet_pos = obs[4:6]
    mallet_vel = obs[6:8]
    distance = np.linalg.norm(puck_pos - mallet_pos)
    puck_push = next_obs[0] - obs[0]
    absorbing = done

    reward = 0
    reward += proximity_to_puck(puck_pos, mallet_pos)
    reward += move_toward_puck(mallet_vel, puck_pos, mallet_pos)
    reward += positioning_penalty(mallet_pos)
    reward += hit_puck(distance)
    reward += small_penalties(distance, mallet_vel)
    reward += behind_puck(puck_pos, mallet_pos)
    reward += push_direction_bonus(puck_push)
    reward += push_direction_penalty(puck_push)
    reward += puck_velocity_bonus(puck_vel)
    reward += puck_alignment(puck_vel)
    reward += urgency_penalty(obs, next_obs, puck_vel, mallet_pos, puck_pos)
    reward += clearance_reward(obs, next_obs, puck_vel)
    reward += goal_event_score(next_obs, absorbing)
    reward += goal_event_conceded(next_obs, absorbing)
    reward += puck_to_goal_velocity(puck_vel, next_obs)
    reward += near_goal_bonus(next_obs)
    reward += stuck_puck_penalty(next_obs, puck_vel)
    reward += velocity_alignment_bonus(puck_pos, mallet_pos, mallet_vel)
    reward += wall_penalty(next_obs)
    reward += direction_change_penalty(obs, next_obs)
    reward += passive_goal_defense(puck_vel, next_obs, mallet_vel)
    reward += puck_trap_bonus(next_obs, puck_pos, mallet_pos)
    reward += minimal_motion_penalty(puck_vel, mallet_vel)
    reward += small_energy_penalty()

    return reward