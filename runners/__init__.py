from typing import Dict

import datasets
import utils
import hooks
from utils.registers import RUNNERS_REG
from . import optimizer
from . import scheduler
from . import base_runner
from . import epoch_runner
from . import iter_runner
from . import utils


def build_runner(args: Dict, config: Dict, model, datasets_dict):
    """
    General build function for runner, config is the FULL version
    """
    _net_dict = utils.distributed.skip_module_for_dict(model.nets)
    optim = optimizer.get_optimizer(
        config.runner.optimizer, config.networks, _net_dict
    )
    sched = scheduler.get_scheduler(
        config.runner.scheduler, optim
    )
    dataloader_dict, sampler_dict = datasets.get_dataloaders(
        config.datasets.dataloaders, datasets_dict, config.misc.seed
    )
    runner = RUNNERS_REG.get(config.runner.type)(
        args, config, model, optim, sched, dataloader_dict, sampler_dict, **config.runner.settings
    )
    return runner
