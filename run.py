import os
import sys
import argparse
import yaml
from munch import Munch, DefaultMunch
import numpy as np
from shutil import copyfile

from model import Trainer, test
from utils.utils import init_seeds, get_logger, select_device, make_save_dir

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
    
    if config.run_mode == 'train':
        train_config = config.train
        train_config.device = config.device
        train_config.random_seed = config.random_seed
        train_config.run_yaml = args.config

        train_config = make_save_dir(train_config)
        train_config.logger = get_logger(train_config)
        train_config.logger.info(f"Result save to : {train_config.save_dir}")
        trainer = Trainer(train_config)
        trainer.run()

    elif config.run_mode == 'test':
        config_test = config.train
        config_test.update(config.test)
        score = test(config_test)
        print('Total score: {0:.2f}'.format(score))
