from typing import Dict

from .base_hook import BaseHook
from . import checkpoint
from . import logger
from . import optimizer
from . import scheduler
from . import metric
from . import model_custom
from . import dataset_custom

from utils.distributed import get_dist_info
from utils.registers import HOOKS_REG


def get_hooks(hook_configs: Dict) -> Dict:
    """General build function for hooks"""
    hook_dict = dict()
    for hook_cfg in hook_configs.hooks:
        # Use hook type as name by default
        # If want to register a hook several times, a name should be given
        if not isinstance(hook_cfg.name, str):
            # if getting from children register, remove scope name
            if '.' in hook_cfg.type:
                hook_name = hook_cfg.type.split('.')[-1]
            else:
                hook_name = hook_cfg.type
        else:
            hook_name = hook_cfg.name
        assert hook_name not in hook_dict

        hook_cls = HOOKS_REG.get(hook_cfg.type)
        # master_only hooks is registered as `None` for rank > 0
        if hasattr(hook_cls, 'master_only'):
            if get_dist_info()[0] != 0:
                hook_dict[hook_name] = None
                continue

        # hook_cfg.settings is an empty dict if not set in config
        hook_dict[hook_name] = hook_cls(**hook_cfg.settings)

    return hook_dict