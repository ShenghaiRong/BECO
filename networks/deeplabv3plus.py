from typing import Dict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

import modules
from modules.convs import DepthwiseSeparableConv
from utils.registers import NETWORKS_REG
from utils.modules import init_weight


@NETWORKS_REG.register_module("deeplabv3plus")
class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decode
    Deeplabv3plus implememts
    This module has five components:

    self.backbone
    self.aspp
    self.projector: an 1x1 conv for lowlevel feature projection
    self.preclassifier: an 3x3 conv for feature mixing, before final classification
    self.classifier: last 1x1 conv for output classification results

    Args:
        backbone: Dict, configs for backbone
        decoder: Dict, configs for decoder

    NOTE: The bottleneck has only one 3x3 conv by default, some implements stack
        two 3x3 convs
    """
    def __init__(self, backbone: Dict, decoder: Dict) -> None:
        super(DeepLabV3Plus, self).__init__()

        self.align_corners = decoder.settings.align_corners
        BN_op = getattr(nn, decoder.settings.norm_layer)
        channels = decoder.settings.channels
        self.backbone = modules.backbones.get_backbone(backbone)
        self.aspp = modules.decoders.get_decoder(decoder)
        self.projector = nn.Sequential( 
            nn.Conv2d(
                decoder.settings.lowlevel_in_channels,
                decoder.settings.lowlevel_channels,
                kernel_size=1, bias=False),
            BN_op(decoder.settings.lowlevel_channels),
            nn.ReLU(inplace=True),
        )
        self.pre_classifier = DepthwiseSeparableConv(
            decoder.settings.norm_layer,
            channels + decoder.settings.lowlevel_channels,
            channels, 3, padding=1
        )

        self.classifier = nn.Conv2d(channels, decoder.settings.num_classes, 1, 1)

        init_weight(self.projector)
        init_weight(self.pre_classifier)
        init_weight(self.classifier)

    def forward(self, x: Tensor) -> Tensor:
        size = (x.shape[2], x.shape[3])
        out = self.backbone(x)
        lowlevel_feature = self.projector(out[1])
        output = self.aspp(out[4])
        output = F.interpolate(
            output,
            size=lowlevel_feature.shape[-2:],
            mode='bilinear', align_corners=self.align_corners
        )
        output = torch.cat([lowlevel_feature, output], dim=1)
        output = self.pre_classifier(output)
        out['embeddings'] = output
        output = self.classifier(output)
        out['pre_logits'] = output
        out['logits'] = F.interpolate(output, size=size, mode='bilinear', align_corners=self.align_corners)

        return out
