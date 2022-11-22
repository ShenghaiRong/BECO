from typing import Dict
from torch import Tensor
from torch import nn

import modules
from utils.registers import NETWORKS_REG
from utils.registers import MODULES_REG


@NETWORKS_REG.register_module("resnet")
class ResNet(nn.Module):
    """
    Returns full ResNet
    """
    def __init__(self, backbone: Dict) -> None:
        super().__init__()

        self.net = modules.backbones.get_backbone(backbone)

    def forward(self, x: Tensor):
        return self.net(x)