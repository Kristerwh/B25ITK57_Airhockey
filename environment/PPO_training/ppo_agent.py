import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def get_action(self, obs):
        dist = self.actor(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_action(self, obs, action):
        dist = self.actor(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs)
        return log_prob, entropy, value
