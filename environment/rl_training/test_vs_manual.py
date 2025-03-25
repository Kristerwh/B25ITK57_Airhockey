from stable_baselines3 import PPO
from airhockey_gym_env import AirHockeyGymEnv

model = PPO.load("ppo_airhockey")
env = AirHockeyGymEnv()
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()

print("Total Reward:", total_reward)

env.close()