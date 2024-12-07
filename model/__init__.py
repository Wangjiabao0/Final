import pdb
import os
import yaml

from munch import Munch, DefaultMunch
import gymnasium as gym
import multiprocessing
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from .utils.parallel_environments import ParallelEnvironments
from .utils.storage import Storage
from .utils.actions import get_action_space, get_actions
from .utils.environment_wrapper import EnvironmentWrapper

def get_model(config):
    model = None
    if config.model_name == 'A2C':
        from .actor_critic import ActorCritic
        model = ActorCritic(config, get_action_space())

    elif config.model_name == 'A2C_DP':
        from .actor_critic_dp import ActorCriticDP
        model = ActorCriticDP(config, get_action_space())

    elif config.model_name == 'A2C_ICM':
        from .actor_critic_icm import ActorCriticICM
        model = ActorCriticICM(config, get_action_space())

    elif config.model_name == 'A2C_NOISE':
        from .actor_critic_noise import ActorCriticNoise
        model = ActorCriticNoise(config, get_action_space())

    elif config.model_name == 'A2C_RE3':
        from .actor_critic_re3 import ActorCriticRE3
        model = ActorCriticRE3(config, get_action_space())
    assert model is not None, f"{config.model_name} is not a valid model"
    return model

class Trainer:
    def __init__(self, config):
        self.model_name = config.model_name
        self.config = config
        self.num_of_processes = min(config.num_of_processes, multiprocessing.cpu_count())
        self.parallel_environments = ParallelEnvironments(
            self.config.stack_size, number_of_processes=self.num_of_processes
        )
        self.model = get_model(config)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)
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

                for step in range(self.config.steps_per_update):
                    # Forward pass
                    probs, log_probs, value, intrinsic_reward = self.model(self.current_observations)
                    actions = get_actions(probs)
                    action_log_probs, entropies = self.compute_action_logs_and_entropies(probs, log_probs)

                    # Interact with environment
                    states, rewards, dones = self.parallel_environments.step(actions)
                    if self.model_name == 'A2C_ICM':
                        intrinsic_reward = self.model.compute_intrinsic_reward(self.current_observations, states, actions)
                    if intrinsic_reward is not None:
                        rewards += intrinsic_reward
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
                _, _, last_values, _ = self.model(self.current_observations)
                expected_rewards = self.storage.compute_expected_rewards(last_values, self.config.reward_discount)
                advantages = expected_rewards - self.storage.values # A = R-V

                # Compute losses
                value_loss = advantages.pow(2).mean()
                entropy_term = self.config.entropy_coef * self.storage.entropies.mean()
                policy_loss = -(advantages * self.storage.action_log_probs).mean()-entropy_term
                loss = policy_loss + self.config.value_loss_coef * value_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                self.optimizer.step()
                self.scheduler.step()

                # Record metrics
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)

                self.record_metrics(update, loss, policy_loss, value_loss, entropy_term)

                # Save model periodically
                if update % self.config.save_frequency == 0 and update > 0:
                    save_path = os.path.join(self.config.save_dir, f'{self.model_name}_{update}.pt')
                    torch.save(self.model.state_dict(), save_path)

        self.writer.close()

    def compute_action_logs_and_entropies(self, probs, log_probs):
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_probs = log_probs.gather(1, indices)
        entropies = -(log_probs * probs).sum(-1)
        return action_log_probs, entropies

    def record_metrics(self, update, loss, policy_loss, value_loss, entropy_term):
        """Record training metrics to TensorBoard."""
        self.writer.add_scalar('Loss/Total', loss.item(), update)
        self.writer.add_scalar('Loss/Policy', policy_loss.item(), update)
        self.writer.add_scalar('Loss/Value', value_loss.item(), update)
        self.writer.add_scalar('Loss/Entropy', entropy_term.item(), update)
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None])
        ).item()
        self.writer.add_scalar('Grad/Total Norm', total_norm, update)

        for param_group in self.optimizer.param_groups:
            self.writer.add_scalar('Learning Rate', param_group['lr'], update)

def eval(config):
    env = gym.make('CarRacing-v2',render_mode='human')
    env_wrapper = EnvironmentWrapper(env, config.stack_size)

    model = get_model(config)
    model.load_state_dict(torch.load(config.pt_path))
    model.eval()
    
    state = env_wrapper.reset()
    state = torch.Tensor([state])
    done = False
    total_score = 0
    while not done:
        probs, _, _, _ = model(state)
        action = get_actions(probs)
        state, reward, done = env_wrapper.step(action[0])
        state = torch.Tensor([state])
        total_score += reward
        env_wrapper.render()
    return total_score
