import pdb
import os
import multiprocessing
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from .utils.parallel_environments import ParallelEnvironments
from .utils.storage import Storage
from .utils.actions import get_action_space, get_actions, compute_action_logs_and_entropies
from .actor_critic import ActorCritic

class InverseForwardDynamicsModel(nn.Module):
    def __init__(self, kwargs):
        super(InverseForwardDynamicsModel, self).__init__()

        self.inverse_model = nn.Sequential(
            nn.Linear(kwargs['latent_dim'] * 2, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, kwargs['action_dim'])
        )

        self.forward_model = nn.Sequential(
            nn.Linear(kwargs['latent_dim'] + kwargs['action_dim'], 64),
            nn.LeakyReLU(),
            nn.Linear(64, kwargs['latent_dim'])
        )
        self.softmax = nn.Softmax()

    def forward(self, obs, action, next_obs, training=True):
        if training:
            # inverse prediction
            im_input_tensor = torch.cat([obs, next_obs], dim=1)
            pred_action = self.inverse_model(im_input_tensor)
            # forward prediction
            fm_input_tensor = torch.cat([obs, action], dim=-1)
            pred_next_obs = self.forward_model(fm_input_tensor)

            return pred_action, pred_next_obs
        else:
            # forward prediction
            fm_input_tensor = torch.cat([obs, action], dim=-1)
            pred_next_obs = self.forward_model(fm_input_tensor)

            return pred_next_obs

class CnnEncoder(nn.Module):
    def __init__(self, obs_shape):
        super(CnnEncoder, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(obs_shape, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.LeakyReLU()
        )

    def forward(self, obs, next_obs=None):
        if next_obs is not None:
            input_tensor = torch.cat([obs, next_obs], dim=1)
        else:
            input_tensor = obs

        latent_vectors = self.main(input_tensor)

        return latent_vectors.view(latent_vectors.size(0), -1)

class ICM(object):
    def __init__(self, config):
        """
        Curiosity-Driven Exploration by Self-Supervised Prediction
        Paper: http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param lr: The learning rate of inverse and forward dynamics model.
        :param batch_size: The batch size to train the dynamics model.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """
        self.device = config.device
        self.beta = config.beta
        self.kappa = config.kappa
        self.lr = config.icm_lr
        self.config = config
        self.batch_size = config.icm_batch_size # TODO

        # Only keep the Box action space case
        self.ob_shape = (config.stack_size, 84, 84)
        self.action_shape = 3
        self.inverse_forward_model = InverseForwardDynamicsModel(
            kwargs={'latent_dim':1024, 'action_dim': self.action_shape}
        ).to(config.device)
        self.im_loss = nn.MSELoss()
        self.fm_loss = nn.MSELoss()
        self.cnn_encoder = CnnEncoder(self.ob_shape[0]).to(config.device)

        self.optimizer = optim.Adam(lr=self.lr, params=self.inverse_forward_model.parameters())

    def update(self, rollouts):
        n_steps = self.config.steps_per_update
        n_envs = self.config.num_of_processes
        obs = torch.stack(rollouts['observations']).reshape(n_steps * n_envs, *self.ob_shape)
        actions = torch.stack(rollouts['actions']).reshape(n_steps * n_envs, self.action_shape)
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        encoded_obs = self.cnn_encoder(obs)

        dataset = TensorDataset(encoded_obs[:-1], actions[:-1], encoded_obs[1:])
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            batch_actions = batch_data[1]
            batch_next_obs = batch_data[2]

            pred_actions, pred_next_obs = self.inverse_forward_model(
                batch_obs, batch_actions, batch_next_obs
            )

            loss = self.im_loss(pred_actions, batch_actions) + \
                   self.fm_loss(pred_next_obs, batch_next_obs)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def compute_irs(self, rollouts, time_steps):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = self.config.steps_per_update
        n_envs = self.config.num_of_processes
        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        obs = torch.stack(rollouts['observations'])
        actions = torch.stack(rollouts['actions'])
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                encoded_obs = self.cnn_encoder(obs[:, idx, :, :, :])
                pred_next_obs = self.inverse_forward_model(
                    encoded_obs[:-1], actions[:-1, idx], next_obs=None, training=False)
                processed_next_obs = torch.clip(encoded_obs[1:], min=-1.0, max=1.0)
                processed_pred_next_obs = torch.clip(pred_next_obs, min=-1.0, max=1.0)

                intrinsic_rewards[:-1, idx] = F.mse_loss(processed_pred_next_obs, processed_next_obs, reduction='mean').cpu().numpy()
            # processed_next_obs = process(encoded_obs[1:n_steps], normalize=True, range=(-1, 1))
            # processed_pred_next_obs = process(pred_next_obs, normalize=True, range=(-1, 1))
        # train the icm
        self.update(rollouts)

        return beta_t * intrinsic_rewards
    
class A2CICMTrainer:
    def __init__(self, config):
        self.model_name = config.model_name
        self.config = config
        self.num_of_processes = min(config.num_of_processes, multiprocessing.cpu_count())
        config.num_of_processes = self.num_of_processes
        self.parallel_environments = ParallelEnvironments(
            self.config.stack_size, number_of_processes=self.num_of_processes
        )
        self.model = ActorCritic(config, get_action_space())

        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.storage = Storage(self.config.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(
            self.num_of_processes, *self.parallel_environments.get_state_shape()
        )
        self.irs = ICM(config)
        self.writer = SummaryWriter(log_dir=self.config.save_dir)
        self.episode_rewards = [[] for _ in range(self.num_of_processes)]

    def run(self):
        num_of_updates = self.config.num_of_steps // self.config.steps_per_update
        self.current_observations = self.parallel_environments.reset()

        print(self.current_observations.size())
        
        with tqdm(total=int(num_of_updates), desc="Training Progress") as pbar:
            for update in range(int(num_of_updates)):
                self.storage.reset_storage()
                samples = {'observations':[], 
                            'actions':[], 
                            'rewards':[],
                            'terminateds':[],
                            'truncateds':[],
                            'next_observations':[]}
        
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
                    samples['actions'].append(torch.tensor(actions).to(torch.float32))
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
                if (update+1) % self.config.save_frequency == 0 and update > 1:
                    save_path = os.path.join(self.config.save_dir, f'{self.model_name}_{update+1}.pt')
                    torch.save(self.model.state_dict(), save_path)

        self.writer.close()
