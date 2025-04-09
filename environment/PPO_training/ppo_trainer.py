import torch
import numpy as np
from collections import deque
from environment.PPO_training.ppo_agent import PPOAgent


class PPOTrainer:
    def __init__(self, obs_dim, action_dim, clip_eps=0.2, gamma=0.99, lam=0.95, lr=3e-4):
        self.agent = PPOAgent(obs_dim, action_dim, lr)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        returns = np.array(advantages) + np.array(values)
        return np.array(advantages), returns

    def ppo_update(self, obs, actions, log_probs_old, returns, advantages, epochs=10, batch_size=64):
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        for _ in range(epochs):
            for i in range(0, len(obs), batch_size):
                batch_slice = slice(i, i + batch_size)
                obs_b = obs[batch_slice]
                act_b = actions[batch_slice]
                old_log_b = log_probs_old[batch_slice]
                ret_b = returns[batch_slice]
                adv_b = advantages[batch_slice]

                new_log, entropy, value = self.agent.evaluate_action(obs_b, act_b)
                ratio = (new_log - old_log_b).exp()
                clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(ratio * adv_b, clip_adv).mean()
                value_loss = ((value - ret_b) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                self.agent.optimizer.zero_grad()
                loss.backward()
                self.agent.optimizer.step()

    def act(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob = self.agent.get_action(obs_tensor)
        return action.squeeze(0).detach().numpy(), log_prob.item()

    def evaluate(self, obs):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            value = self.agent.critic(obs_tensor)
        return value.item()

    def save(self, path):
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.agent.actor.state_dict(), path + "_actor.pt")
        torch.save(self.agent.critic.state_dict(), path + "_critic.pt")

    def load(self, path):
        self.agent.actor.load_state_dict(torch.load(path + "_actor.pt"))
        self.agent.critic.load_state_dict(torch.load(path + "_critic.pt"))