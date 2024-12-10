

import os
import pdb
import os
import yaml

from munch import Munch, DefaultMunch
import gymnasium as gym
from tqdm import tqdm
import torch
import sys
import argparse
import yaml
from munch import Munch, DefaultMunch

from utils.utils import init_seeds, get_logger, select_device, make_save_dir

def get_trainer(config):
    trainer = None
    model_yaml_path = f'cfg/models/{config.model_name}.yaml'
    with open(model_yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    model_config = Munch(params)
    model_config = DefaultMunch.fromDict(model_config.toDict())
    model_config.update(config)
    if config.model_name == 'A2C':
        trainer = actor_critic.A2CTrainer(model_config)
    elif config.model_name == 'A2C_DP':
        trainer = actor_critic_dp.A2CDPTrainer(model_config)
    elif config.model_name == 'A2C_ICM':
        trainer = actor_critic_icm.A2CTrainer(model_config)
    elif config.model_name == 'A2C_NOISE':
        trainer = actor_critic_noise.ActorCriticNoise(model_config)
    elif config.model_name == 'A2C_RE3':
        trainer = actor_critic_re3.A2CTrainer(model_config)    
    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[DRL] Train the car racing model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    args = parser.parse_args()
    if not args.config:
        raise ValueError("Configuration file must be provided using --config")
    with open(args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    config = Munch(params, **vars(args))
    config = DefaultMunch.fromDict(config.toDict())

    init_seeds(config.random_seed)
    config.device = select_device(config.device)

    env = gym.make('CarRacing-v2',render_mode='human')
    env_wrapper = EnvironmentWrapper(env, config.stack_size)

    model = get_model(config)
    pt = torch.load(config.pt_path)
    model.load_state_dict(pt)
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
