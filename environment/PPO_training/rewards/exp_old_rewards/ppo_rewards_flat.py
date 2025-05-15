
from environment.PPO_training.rewards.exp_old_rewards.reward_utils import (
    proximity_to_puck,
    move_toward_puck,
    hit_puck,
    push_right,
    push_wrong,
    puck_speed,
    puck_direction,
    passive_penalty,
    behind_puck,
    overposition,
    goal_scored,
    goal_conceded
)

def flat_reward(obs, action, next_obs, done, env, ep):
    reward = 0
    reward += proximity_to_puck(obs, action, next_obs, done, env, ep)
    reward += move_toward_puck(obs, action, next_obs, done, env, ep)
    reward += hit_puck(obs, action, next_obs, done, env, ep)
    reward += push_right(obs, action, next_obs, done, env, ep)
    reward += push_wrong(obs, action, next_obs, done, env, ep)
    reward += puck_speed(obs, action, next_obs, done, env, ep)
    reward += puck_direction(obs, action, next_obs, done, env, ep)
    reward += passive_penalty(obs, action, next_obs, done, env, ep)
    reward += behind_puck(obs, action, next_obs, done, env, ep)
    reward += overposition(obs, action, next_obs, done, env, ep)
    reward += goal_scored(obs, action, next_obs, done, env, ep)
    reward += goal_conceded(obs, action, next_obs, done, env, ep)
    return reward
