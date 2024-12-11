import pdb
import os
import multiprocessing
from tqdm import tqdm
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from .utils.parallel_environments import ParallelEnvironments
from .utils.storage import Storage
from .utils.actions import get_action_space, get_actions, compute_action_logs_and_entropies

class NoisyLinear(nn.Linear):
  '''
  https://github.com/Kaixhin/NoisyNet-A3C.git
  '''
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
    self.sigma_init = sigma_init
    self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
    self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only nn.init after all params added (otherwise super().__init__() fails)
      nn.init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      nn.init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      nn.init.constant_(self.sigma_weight, self.sigma_init)
      nn.init.constant_(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * torch.autograd.Variable(self.epsilon_weight), self.bias + self.sigma_bias * torch.autograd.Variable(self.epsilon_bias))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)
    
class ActorCriticNoise(nn.Module):
    def __init__(self, config, num_of_actions):
        super().__init__()
        num_of_inputs = config.stack_size
        sigma_init = config.sigma_init
        self.conv1 = nn.Conv2d(num_of_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = nn.Linear(32 * 9 * 9, 256) 

        self.policy = NoisyLinear(256, num_of_actions, sigma_init=sigma_init)
        self.value = NoisyLinear(256, 1, sigma_init=sigma_init) 

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        flattened = torch.flatten(conv2_out, start_dim=1)
        linear1_out = self.linear1(flattened)
        policy_output = self.policy(linear1_out)
        value_output = self.value(linear1_out)
        probs = F.softmax(policy_output, dim=1)
        log_probs = F.log_softmax(policy_output, dim=1)
        return probs, log_probs, value_output
    
    def sample_noise(self): # train
        self.policy.sample_noise()
        self.value.sample_noise()

    def remove_noise(self): # for test
        self.policy.remove_noise()
        self.value.remove_noise()


class A2CNoiseTrainer:
    def __init__(self, config):
        self.model_name = config.model_name
        self.config = config
        self.num_of_processes = min(config.num_of_processes, multiprocessing.cpu_count())
        self.parallel_environments = ParallelEnvironments(
            self.config.stack_size, number_of_processes=self.num_of_processes
        )
        self.model = ActorCriticNoise(config, get_action_space())
        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.storage = Storage(self.config.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(
            self.num_of_processes, *self.parallel_environments.get_state_shape()
        )
        self.writer = SummaryWriter(log_dir=self.config.save_dir)
        self.episode_rewards = [[] for _ in range(self.num_of_processes)]

    def run(self):
        num_of_updates = self.config.num_of_steps // self.config.steps_per_update
        self.current_observations = self.parallel_environments.reset()
        with tqdm(total=int(num_of_updates), desc="Training Progress") as pbar:
            for update in range(int(num_of_updates)):
                self.storage.reset_storage()
                self.model.sample_noise() # Pick a new noise vector (until next optimisation step)

                for step in range(self.config.steps_per_update):
                    # Forward pass
                    probs, log_probs, value = self.model(self.current_observations)
                    actions = get_actions(probs)
                    action_log_probs, entropies = compute_action_logs_and_entropies(probs, log_probs)

                    # Interact with environment
                    states, rewards, dones = self.parallel_environments.step(actions)
                    rewards = rewards.view(-1, 1)
                    dones = dones.view(-1, 1)

                    # Accumulate rewards for each process
                    for i in range(self.num_of_processes):
                        self.episode_rewards[i].append(rewards[i].item())
                        if dones[i]:
                            self.writer.add_scalar(f'Reward/Episode Reward - Process {i}', sum(self.episode_rewards[i]), update)
                            self.episode_rewards[i] = []

                    # Store experiences
                    self.current_observations = states
                    self.storage.add(step, value, rewards, action_log_probs, entropies, dones)

                # Compute R and V
                _, _, last_values = self.model(self.current_observations)
                expected_rewards = self.storage.compute_expected_rewards(last_values, self.config.reward_discount)
                advantages = expected_rewards - self.storage.values # A = R-V
                self.writer.add_scalar('rewards', expected_rewards.mean(), update)

                # Compute losses
                value_loss = advantages.pow(2).mean()
                entropy_term = self.config.entropy_coef * self.storage.entropies.mean()
                policy_loss = -(advantages * self.storage.action_log_probs).mean()-entropy_term
                loss = policy_loss + self.config.value_loss_coef * value_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad, 2)
                self.optimizer.step()
                # Record metrics
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)

                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None])
                ).item()
                self.writer.add_scalar('Grad/Total Norm', total_norm, update)

                # Save model periodically
                if update % self.config.save_frequency == 0 and update > 0:
                    save_path = os.path.join(self.config.save_dir, f'{self.model_name}_{update}.pt')
                    torch.save(self.model.state_dict(), save_path)

        self.writer.close()
