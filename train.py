import os
import sys
import argparse
import yaml
from shutil import copyfile
from munch import Munch, DefaultMunch
import numpy as np
from model import actor_critic, actor_critic_icm, actor_critic_dp, actor_critic_noise, actor_critic_re3
from utils.utils import init_seeds, get_logger, select_device, make_save_dir

def get_trainer(config):
    trainer = None
    model_yaml_path = f'cfg/models/{config.model_name}.yaml'
    with open(model_yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    model_config_file = os.path.basename(model_yaml_path)
    copyfile(model_yaml_path, os.path.join(config.save_dir, model_config_file))
    model_config = Munch(params)
    model_config = DefaultMunch.fromDict(model_config.toDict())
    model_config.update(config)
    if config.model_name == 'A2C':
        trainer = actor_critic.A2CTrainer(model_config)
    elif config.model_name == 'A2C_DP':
        trainer = actor_critic_dp.A2CDPTrainer(model_config)
    elif config.model_name == 'A2C_ICM':
        trainer = actor_critic_icm.A2CICMTrainer(model_config)
    elif config.model_name == 'A2C_NOISE':
        trainer = actor_critic_noise.A2CNoiseTrainer(model_config)
    elif config.model_name == 'A2C_RE3':
        trainer = actor_critic_re3.A2CRE3Trainer(model_config)    
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

    train_config = make_save_dir(config)
    train_config.logger = get_logger(config)
    train_config.logger.info(f"Result save to : {config.save_dir}")
    trainer = get_trainer(config)
    trainer.run()
