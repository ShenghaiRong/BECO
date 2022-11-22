import os
import random
import platform
import functools
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from addict import Dict
import pickle
import logging

def init_dist():
    """
    Setting up DDP environment. To simplifying the process, this framework does
    noe support DDP over multiple machines
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randint(30000, 65535))


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_dist_info():
    if is_distributed():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_device():
    if is_distributed():
        rank = dist.get_rank()
        device = torch.device('cuda', rank)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def get_ddp_net(networks_dict: Dict):
    """Wrap network dict in DDP"""
    if is_distributed():
        for k, v in networks_dict.items():
            # Check if this network requires update
            if next(v.parameters()).requires_grad:
                networks_dict[k] = DistributedDataParallel(v, 
                                                device_ids=[dist.get_rank()])
            else:
                pass
    return networks_dict

def skip_module(net) -> nn.Module:
    """Skip `module` in DDP wrapped net"""
    if hasattr(net, 'module'):
        return net.module
    else:
        return net

def skip_module_for_dict(net_dict: Dict) -> Dict:
    """Skip `module` in DDP wrapped net dict"""
    _net_dict = dict()
    for k, v in net_dict.items():
       _net_dict[k] = skip_module(v)
    return _net_dict 

def master_only(func):
    """master_only decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []