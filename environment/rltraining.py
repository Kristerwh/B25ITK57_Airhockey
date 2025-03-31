from environment.env_settings.environments.iiwas.env_base import AirHockeyBase
from tqdm import trange
import numpy as np
from environment.rlagent import RLAgent

env = AirHockeyBase()
input_shape = len(env.get_all_observation_keys())
action_dim = 2

agent = RLAgent(input_shape)
agent.compile("adam", "mean_squared_error", "mean_absolute_error")

episodes = 5

for episode in trange(episodes, desc="Training Episodes"):
    obs = env.reset()
    action = agent.predict(obs)
    total_reward = 0

    for _ in range(env._mdp_info.horizon):
        action = agent.predict(obs)
        next_obs, reward, absorbing, _ = env.step(action)
        agent.fit(np.array([obs]), np.array([action]), 1)  # supervised-style

        obs = next_obs
        total_reward += reward

        if absorbing:
            break

    print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")