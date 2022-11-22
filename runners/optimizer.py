from typing import Dict, List, Union, Iterable

from torch import ne, optim
import torch.nn as nn

from utils.misc import rgetattr

from fnmatch import fnmatch


def get_optimizer(optim_config: List, net_configs: List, net_dict: Dict):
    """
    Network is registered one-to-one to coresponding optimizer. 

    # NOTE: If multiple networks are presented, The designed is using a single
        optimizer to opertate all theses networks. This implementation has not
        be validated before
    """
    def build_param_from_cfg(net_configs, optim_config):
        param_group_list = list()
        for net_cfg in net_configs:
            net_name = net_cfg.name
            optim_cfg = net_cfg.optimizer
            #param_group_cfg = net_cfg.param_group
            # Not use param group
            if 'param_groups' not in net_cfg:
                param_group_list.append(
                    dict(params = net_dict[net_name].parameters())
                )
            # Not use optimizer
            #if param_group_cfg is None:
            #    continue
            # Use param group, the optim_cfg is a list of Dict
            else:
                param_groups = get_parameter_group(
                                    net_dict[net_name],
                                    optim_config.settings.lr,
                                    optim_config.settings.weight_decay,
                                    net_cfg.param_groups)
                param_group_list = param_groups

        return param_group_list

    try:
        optim_cls = getattr(optim, optim_config.type)
    except Exception as e:
        raise RuntimeError(f"Error occured when trying to find optimizer {optim_config.type}")\

    param_group_list = build_param_from_cfg(net_configs, optim_config)
    # If no param needs optimizing, return None
    return optim_cls(param_group_list, **optim_config.settings) \
        if len(param_group_list) > 0 else None


def get_parameter_group(
        model: nn.Module,
        base_lr: float,
        base_weight_decay: float,
        groups: Dict[str, float],
        ignore_the_rest: bool = False,
        raw_query: bool = False
        ) -> List[Dict[str, Union[float, Iterable]]]:
        """Fintune.
        """
        for query, lr in groups.items():
            assert isinstance(lr, list), 'the query of param_groups must be a list'
            if len(lr) == 1:
                groups[query].append(1)

        parameters = [
          dict(params=[], names=[], query=query if raw_query else '*' + query + '*',
               lr=lr_wd[0] * base_lr, weight_decay=lr_wd[1] * base_weight_decay) for query, lr_wd in groups.items()]
        rest_parameters = dict(params=[], names=[], lr=base_lr, weight_decay=base_weight_decay)
        for k, v in model.named_parameters():
          matched = False
          for group in parameters:
            if fnmatch(k, group['query']):
              group['params'].append(v)
              group['names'].append(k)
              matched = True
              break
          if not matched:
            rest_parameters['params'].append(v)
            rest_parameters['names'].append(k)
        if not ignore_the_rest:
          parameters.append(rest_parameters)
        for group in parameters:
          group['params'] = iter(group['params'])
        return parameters


def get_optimizer_old(optim_config: List, net_configs: List, net_dict: Dict):
    """
    Network is registered one-to-one to coresponding optimizer. 

    # NOTE: If multiple networks are presented, The designed is using a single
        optimizer to opertate all theses networks. This implementation has not
        be validated before
    """
    def build_param_from_cfg(net_configs):
        param_group_list = list()
        for net_cfg in net_configs:
            net_name = net_cfg.name
            optim_cfg = net_cfg.optimizer
            # Not use optimizer
            if optim_cfg is None:
                continue
            # Not use param group
            if len(optim_cfg) == 0:
                param_group_list.append(
                    dict(params = net_dict[net_name].parameters())
                )
            # Use param group, the optim_cfg is a list of Dict
            else:
                for item in optim_cfg:
                    param_name = item.pop('params')
                    item.params = rgetattr(net_dict[net_name], param_name).parameters()
                    param_group_list.append(item)

        return param_group_list

    try:
        optim_cls = getattr(optim, optim_config.type)
    except Exception as e:
        raise RuntimeError(f"Error occured when trying to find optimizer {optim_config.type}")\

    param_group_list = build_param_from_cfg(net_configs)
    # If no param needs optimizing, return None
    return optim_cls(param_group_list, **optim_config.settings) \
        if len(param_group_list) > 0 else None