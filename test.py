import pdb
import yaml
import os

from munch import Munch, DefaultMunch
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import torch
import argparse
import yaml
from munch import Munch, DefaultMunch

from model import actor_critic, actor_critic_icm, actor_critic_dp, actor_critic_noise, actor_critic_re3
from utils.utils import init_seeds, get_logger, select_device, make_save_dir
from model.utils import environment_wrapper, actions


def get_model(config):
    model = None
    model_yaml_path = f'cfg/models/{config.model_name}.yaml'
    with open(model_yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    model_config = Munch(params)
    model_config = DefaultMunch.fromDict(model_config.toDict())
    model_config.update(config)
    if config.model_name == 'A2C':
        model = actor_critic.ActorCritic(model_config, actions.get_action_space())
    elif config.model_name == 'A2C_DP':
        model = actor_critic_dp.ActorCriticDP(model_config, actions.get_action_space())
    elif config.model_name == 'A2C_ICM':
        model = actor_critic_icm.ActorCritic(model_config, actions.get_action_space())
    elif config.model_name == 'A2C_NOISE':
        model = actor_critic_noise.ActorCriticNoise(model_config, actions.get_action_space())
    elif config.model_name == 'A2C_RE3':
        model = actor_critic_re3.ActorCritic(model_config, actions.get_action_space())    
    return model


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

    if config.save_video:
        env = gym.make('CarRacing-v2',render_mode='rgb_array')

        video_save_folder = os.path.join(os.path.dirname(config.pt_path),'video')
        video_name = os.path.basename(config.pt_path)[:-3]
        env = RecordVideo(env, video_folder=video_save_folder, 
                          episode_trigger=lambda x: True, name_prefix=video_name)
        
    else:
        env = gym.make('CarRacing-v2',render_mode='human')

    env_wrapper = environment_wrapper.EnvironmentWrapper(env, config.stack_size)

    model = get_model(config)
    pt = torch.load(config.pt_path)
    model.load_state_dict(pt)
    model.eval()
    
    state = env_wrapper.reset(seed=config.random_seed)
    state = torch.Tensor([state])
    done = False
    total_score = 0
    while not done:
        probs, _, _, = model(state)
        action = actions.get_actions(probs)
        state, reward, done = env_wrapper.step(action[0])
        state = torch.Tensor([state])
        total_score += reward
        env_wrapper.render()
    print('score is ', total_score)