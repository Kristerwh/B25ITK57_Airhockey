from stable_baselines3 import PPO
from airhockey_gym_env import AirHockeyGymEnv

env = AirHockeyGymEnv()

obs, _ = env.reset()

model = PPO("MlpPolicy", env, verbose=1)

#Start training
model.learn(total_timesteps=200_000)

#model save
model.save("ppo_airhockey")
print("Training complete.")
