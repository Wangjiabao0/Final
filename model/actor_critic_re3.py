import pdb
import os
import multiprocessing

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .actor_critic import ActorCritic
from .utils.parallel_environments import ParallelEnvironments
from .utils.storage import Storage
from .utils.actions import get_action_space, get_actions, compute_action_logs_and_entropies


class CnnEncoder(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super(CnnEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, (8, 8), stride=(4, 4)), nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2)), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1)), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim), nn.LayerNorm(latent_dim))
    def forward(self, ob):
        x = self.main(ob)
        return x

class RE3(object):
    def __init__(self, config, action_shape):
        """
        State Entropy Maximization with Random Encoders for Efficient Exploration (RE3)
        Paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf

        :param obs_shape: The data shape of observations.
        :param action_shape: The data shape of actions.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        self.obs_shape = (config.stack_size, 84, 84) # with bathsize process numbers
        self.action_shape = action_shape
        self.device = config.device
        self.beta = config.beta
        self.kappa = config.kappa

        self.config = config

        self.encoder = CnnEncoder(self.obs_shape, config.latent_dim)

        self.encoder.to(self.device)

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def compute_irs(self, rollouts, time_steps, k=3, average_entropy=True):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :param k: The k value.
        :param average_entropy: Use the average of entropy estimation.
        :return: The intrinsic rewards
        """
        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = self.config.steps_per_update
        n_envs = self.config.num_of_processes
        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = torch.stack(rollouts['observations'])
        obs_tensor = obs_tensor.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.encoder(obs_tensor[:, idx])
                dist = torch.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
                if average_entropy:
                    for sub_k in range(k):
                        intrinsic_rewards[:, idx, 0] += torch.log(
                            torch.kthvalue(dist, sub_k + 1, dim=1).values + 1.).cpu().numpy()
                    intrinsic_rewards[:, idx, 0] /= k
                else:
                    intrinsic_rewards[:, idx, 0] = torch.log(
                            torch.kthvalue(dist, k + 1, dim=1).values + 1.).cpu().numpy()
        
        return beta_t * intrinsic_rewards
    
class A2CRE3Trainer:
    def __init__(self, config):
        self.model_name = config.model_name
        self.config = config
        # set subprocesses
        self.num_of_processes = min(config.num_of_processes, multiprocessing.cpu_count())
        config.num_of_processes = self.num_of_processes
        self.parallel_environments = ParallelEnvironments(self.config.stack_size, 
                                                          number_of_processes=self.num_of_processes)
        self.model = ActorCritic(config, get_action_space())
        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.storage = Storage(self.config.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(self.num_of_processes, 
                                                *self.parallel_environments.get_state_shape())
        self.irs = RE3(config, get_action_space())
        self.writer = SummaryWriter(log_dir=self.config.save_dir)
        self.episode_rewards = [[] for _ in range(self.num_of_processes)]

    def run(self):
        num_of_updates = self.config.num_of_steps // self.config.steps_per_update
        self.current_observations = self.parallel_environments.reset()
        print(self.current_observations.size())
        
        with tqdm(total=int(num_of_updates), desc="Training Progress") as pbar:
            for update in range(int(num_of_updates)):
                samples = {'observations':[], 
                            'actions':[], 
                            'rewards':[],
                            'terminateds':[],
                            'truncateds':[],
                            'next_observations':[]}
                self.storage.reset_storage()
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
                    samples['observations'].append(self.current_observations)
                    samples['actions'].append(actions)
                    samples['rewards'].append(rewards)
                    samples['terminateds'].append(dones)
                    samples['next_observations'].append(states)
                    self.current_observations = states
                    self.storage.add(step, value, rewards, action_log_probs, entropies, dones)

                intrinsic_rewards = torch.from_numpy(self.irs.compute_irs(samples, update)).to(torch.float32)
                # Compute R and V
                _, _, last_values = self.model(self.current_observations)
                expected_rewards = self.storage.compute_expected_rewards(last_values, self.config.reward_discount)
                rewards = expected_rewards + intrinsic_rewards
                advantages = rewards - self.storage.values # A = R-V
                self.writer.add_scalar('intrinsic_rewards', intrinsic_rewards.mean(), update)
                self.writer.add_scalar('expected_rewards', expected_rewards.mean(), update)
                self.writer.add_scalar('rewards', rewards.mean(), update)

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
