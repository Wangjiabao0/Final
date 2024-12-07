import torch
import torch.nn as nn
import torch.nn.functional as F
from .actor_critic import ActorCritic

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=256):
        super(ICM, self).__init__()
        # Feature Encoder
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU()
        )

        # Inverse Dynamics Model
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # Forward Dynamics Model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state, next_state, action):
        # Encode features
        phi_state = self.feature_encoder(state)
        phi_next_state = self.feature_encoder(next_state)

        # Inverse Model: Predict action
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        predicted_action = self.inverse_model(inverse_input)

        # Forward Model: Predict next state features
        forward_input = torch.cat([phi_state, action], dim=1)
        predicted_phi_next_state = self.forward_model(forward_input)

        return phi_state, phi_next_state, predicted_action, predicted_phi_next_state

    def compute_intrinsic_reward(self, phi_next_state, predicted_phi_next_state):
        # Intrinsic reward as prediction error
        return 0.5 * ((phi_next_state - predicted_phi_next_state) ** 2).sum(dim=1)

class ActorCriticICM(nn.Module):
    def __init__(self, config, num_of_actions):
        super().__init__()
        num_of_inputs = config.stack_size
        self.actor_critic = ActorCritic(config, num_of_actions)
        self.icm = ICM(num_of_inputs, num_of_actions)

    def forward(self, state):
        probs, log_probs, value, _ = self.actor_critic(state)

        return probs, log_probs, value, None
    
    def compute_intrinsic_reward(self, state, next_state, action):
        phi_state, phi_next_state, predicted_action, predicted_phi_next_state = self.icm(state, next_state, action)
        intrinsic_reward = self.icm.compute_intrinsic_reward(phi_next_state, predicted_phi_next_state)
        return intrinsic_reward
