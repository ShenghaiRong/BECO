from typing import Dict

from utils.registers import MODULES_REG
from . import resnet
from . import mit


def get_backbone(config: Dict):
    backbone_obj = MODULES_REG.BACKBONES.get(config.type)
    return backbone_obj(config.pretrain, **config.settings)
