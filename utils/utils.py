import os
import random
from shutil import copyfile
import logging
import datetime

import numpy as np
import torch
import cv2
from pynvml import (
    nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo, 
    nvmlDeviceGetName,  nvmlShutdown, nvmlDeviceGetCount
)

def cuda_mem(config):
    fill = 0
    n = datetime.now()
    nvmlInit()
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle)
        config.logger.info("[ {} ]-[ GPU{}: {}".format(n, 0, gpu_name))
        config.logger.info("total GPU memory: {:.3}G".format((info.total // 1048576) / 1024))
        config.logger.info("free GPU memory: {:.3}G".format((info.free // 1048576) / 1024))
        model_use = (info.used  // 1048576) - fill
        config.logger.info("model used GPU memory: {:.3}G({}MiB)".format( model_use / 1024, model_use))
    nvmlShutdown()

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    torch.set_default_dtype(torch.float)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_save_dir(config):
    from datetime import datetime
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    # Make save directory
    save_dir = os.path.join(
        config.save_dir, config.model_name + "_" + str(current_time))
    os.makedirs(save_dir)
    config.save_dir = save_dir
    # Configuration file save
    config_file = os.path.basename(config.run_yaml)
    copyfile(config.run_yaml, os.path.join(save_dir, config_file))
    
    return config

def get_logger(config, stream=True):
    
    logger = logging.getLogger(__name__)
    format = logging.Formatter(
        '[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> [DRL] %(message)s')

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(format)
        logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(config.save_dir, 'result.log'))
    file_handler.setFormatter(format)

    logger.addHandler(file_handler)
    logger.setLevel(level=logging.DEBUG)

    return logger

def select_device(device='', batch_size=None, only_visible=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'[DRL] inference framework, torch {torch.__version__} '  # string
    cpu = str(device).lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device and only_visible:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
        
    return torch.device('cuda:0' if cuda else 'cpu')