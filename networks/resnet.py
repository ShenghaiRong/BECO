from typing import Dict
from torch import Tensor
from torch import nn

import modules


class ResNet(nn.Module):
    """
    Returns full ResNet
    """
    def __init__(self, backbone: Dict) -> None:
        super().__init__()

        self.net = modules.backbones.get_backbone(backbone)

    def forward(self, x: Tensor):
        return self.net(x)