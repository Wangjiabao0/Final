import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import deque
from .actor_critic import ActorCritic

class RandomEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim=50):
        super(RandomEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, latent_dim),  # Adjust size based on input dimensions
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )
        for param in self.parameters():
            param.requires_grad = False  # Freeze the encoder weights

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class RE3:
    def __init__(self, encoder, k=3, buffer_size=500):
        self.encoder = encoder
        self.k = k
        self.replay_buffer = deque(maxlen=buffer_size)

    def add_to_buffer(self, states):
        with torch.no_grad():
            representations = self.encoder(states).cpu().numpy()
        self.replay_buffer.extend(representations)
        print (len(self.replay_buffer))
    
    def reset_buffer(self):
        """清空 replay_buffer"""
        self.replay_buffer.clear()

    def compute_intrinsic_reward(self, states):
        if len(self.replay_buffer) < self.k:
            return torch.zeros(states.size(0))

        with torch.no_grad():
            representations = self.encoder(states).cpu().numpy()

        # k-NN entropy estimation
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(self.replay_buffer)
        distances, _ = nbrs.kneighbors(representations)

        # Compute intrinsic reward
        intrinsic_rewards = -np.log(distances.mean(axis=1) + 1e-8)
        return torch.tensor(intrinsic_rewards, dtype=torch.float32)

class ActorCriticRE3(nn.Module):
    def __init__(self, config, num_of_actions):
        super().__init__()
        num_of_inputs = config.stack_size
        k = config.k_neighbors
        self.actor_critic = ActorCritic(config, num_of_actions)
        self.encoder = RandomEncoder(num_of_inputs, latent_dim=config.latent_dim)
        self.re3 = RE3(self.encoder, k, config.buffer_size)

    def forward(self, x):
        probs, log_probs, value, _ = self.actor_critic(x)
        intrinsic_reward = self.re3.compute_intrinsic_reward(x)
        return probs, log_probs, value, intrinsic_reward
