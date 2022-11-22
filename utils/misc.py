import os
import random
import functools

import torch
from torch.distributed import get_rank
from utils.distributed import is_distributed
import numpy as np
from scipy.optimize import curve_fit
import imageio

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


def curve_func(x, a, b, c):
    return a * (1 - np.exp(-1 / c * x ** b))


def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 1, 1), 
                           method='trf', sigma=np.geomspace(1, .1, len(y)),
                           absolute_sigma=True, 
                           bounds=([0, 0, 0], [1, 1, np.inf]))
    return tuple(popt)


def derivation(x, a, b, c):
    x = x + 1e-6  # numerical robustness
    return a * b * 1 / c * np.exp(-1 / c * x ** b) * (x ** (b - 1))


def label_update_epoch(ydata_fit, threshold=0.9, end=1, max_epoch=16):
    xdata_fit = np.linspace(0, end, len(ydata_fit), endpoint=False)
    a, b, c = fit(curve_func, xdata_fit, ydata_fit)
    #epoch = np.arange(1, max_epoch)
    epoch = np.arange(1, max_epoch+1)
    y_hat = curve_func(epoch, a, b, c)
    numerator = abs(abs(derivation(epoch, a, b, c))-abs(derivation(1, a, b, c)))
    denominator = abs(derivation(1, a, b, c))
    relative_change = numerator / denominator
    relative_change[relative_change > 1] = 0
    update_epoch = np.sum(relative_change <= threshold) + 1
    current_epoch_der = relative_change[end-1]
    return update_epoch, current_epoch_der
        

def if_update(iou_value, current_epoch, max_epoch, threshold):
    update_epoch, _ = label_update_epoch(iou_value, threshold, 
                                      current_epoch, max_epoch)
    return current_epoch >= update_epoch  # , update_epoch 

def if_update_with_der(iou_value, current_epoch, max_epoch, threshold):
    update_epoch, derivat = label_update_epoch(iou_value, threshold, 
                                      current_epoch, max_epoch)
    return current_epoch >= update_epoch, derivat  # , update_epoch 


def get_logging_path(cfg):
    try:
        logging_path = cfg.hooks['hooks'][0]['settings']['path']
    except ModuleNotFoundError as e:
        print('logging_path is not found in config.hooks')
    else:
        return logging_path
    




    