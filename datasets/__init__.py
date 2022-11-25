from functools import partial
from typing import Dict, List
import random
import importlib

import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

from datasets.transforms import transform
from datasets.transforms.compose import Compose
from utils import distributed

from . import voc
from . import coco



# For Reproducibility
def workder_init_fn(worker_id, num_workers, rank, usr_seed):
    worker_seed = num_workers * rank + worker_id + usr_seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(config: Dict, datasets_dict: Dict, seed: int) -> Dict:
    """
    General build function for dataloaders

    TODO: Add collate_fn support

    NOTE: if loading a very large dataset, might replace Python lists with
    non-refcounted representations such as Pandas, Numpy or PyArrow objects.
    See https://github.com/pytorch/pytorch/issues/13246 for more details

    """

    rng = torch.Generator()
    rng.manual_seed(seed)
    rank, world_size = distributed.get_dist_info()
    init_fn = partial(
        workder_init_fn, num_workers = config['num_workers'],
        rank = rank, usr_seed = seed   
    )

    # Prepare dataloader sampler for DDP
    sampler_dict = dict()
    if distributed.is_distributed():
        for k in datasets_dict.keys():
            is_train = (k == "train")
            sampler_dict[k] = DistributedSampler(
                datasets_dict[k], world_size, rank,
                shuffle=is_train, seed=seed,
                drop_last=config['drop_last'] if is_train else False
            )
    else:
        for k in datasets_dict.keys():
            sampler_dict[k] = None

    # Preparing dataloader
    loader_dict = dict()
    for k in datasets_dict.keys():
        is_train = (k == "train")
        loader_dict[k] = DataLoader(
            dataset=datasets_dict[k],
            batch_size=config['batch_size'] if is_train else 1,
            shuffle=(sampler_dict[k] is None and is_train),
            sampler=sampler_dict[k],
            num_workers=config['num_workers'],
            drop_last=config['drop_last'],
            pin_memory=True,
            worker_init_fn = init_fn,
            generator = rng,
            persistent_workers=True
        )

    return loader_dict, sampler_dict


def get_transforms(config: List, method: str):
    """
    General build function for transforms
    """
    trans_list = list()
    for item in config:
        trans = getattr(transform, item.type)
        if not hasattr(item, 'settings'):
            trans_list.append(trans())
        else:
            trans_list.append(trans(**item.settings))

    return Compose(trans_list)


def get_classmap(dataset_type: str):
    if "VOC" in dataset_type:
        return voc.classes_map
    else:
        raise ValueError