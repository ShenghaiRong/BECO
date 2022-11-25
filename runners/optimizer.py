from typing import Dict, List, Union, Iterable

from torch import optim



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
            net_name = net_cfg['name']

            param_group_list.append(
                    dict(params = net_dict[net_name].parameters())
                )


        return param_group_list

    try:
        optim_cls = getattr(optim, optim_config['type'])
    except Exception as e:
        raise RuntimeError(f"Error occured when trying to find optimizer {optim_config['type']}")\

    param_group_list = build_param_from_cfg(net_configs, optim_config)
    # If no param needs optimizing, return None
    return optim_cls(param_group_list, **optim_config['settings']) \
        if len(param_group_list) > 0 else None


