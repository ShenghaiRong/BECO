import os
import random
import functools

import torch
from torch.distributed import get_rank
from utils.distributed import is_distributed
import numpy as np

"""
Utilities that does not belong to any categories can be put here
"""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_cuda(args):
    torch.backends.cudnn.deterministic = args.determ
    torch.backends.cudnn.benchmark = not args.determ
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if is_distributed():
        torch.cuda.set_device(get_rank())

def rgetattr(obj, path):
    try:
        return functools.reduce(getattr, path.split('.'), obj)
    except AttributeError:
        raise AttributeError(f"{path} cannot be found in {obj.__name__}")



def get_logging_path(cfg):
    try:
        logging_path = cfg.hooks['hooks'][0]['settings']['path']
    except ModuleNotFoundError as e:
        print('logging_path is not found in config.hooks')
    else:
        return logging_path
    




    