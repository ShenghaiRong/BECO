from typing import List
from copy import deepcopy

from utils.registers import NETWORKS_REG
from utils.distributed import get_device
from utils.modules import freeze, unfreeze
from utils.misc import rgetattr

from . import deeplabv2
from . import deeplabv3
from . import deeplabv3plus
from . import resnet
from . import segformer


def get_network(config: List):
    device = get_device()
    network_dict = dict()
    for item in config:
        # Prevent changing original configs
        _item = deepcopy(item)
        # Pop network irrelated k-v
        name = _item.pop("name")
        optimizer_op = _item.pop("optimizer")
        freeze_op = _item.pop("freeze")
        if 'param_groups' in _item:
            param_group = _item.pop("param_groups")
        
        # Get network and send to device
        net_cls = NETWORKS_REG.get(_item.pop("type"))
        net = net_cls(**_item)
        net = net.to(device)

        # If optimizer is none, full network is freezed
        if optimizer_op is None:
            freeze(net)
        # If optimizer is specified, only the param group is not freezed
        elif len(optimizer_op) > 0:
            # First freeze the full network, then unfreeze the given parts
            freeze(net)
            for param_group in optimizer_op:
                module = rgetattr(net, param_group['params'])
                unfreeze(module)

        # Freeze operation works independently with optimizer configs and
        # has higher priority
        if freeze_op is not None:
            for mod_name in freeze_op:
                module = rgetattr(net, mod_name)
                freeze(module)

        network_dict[name] = net
        
    return network_dict