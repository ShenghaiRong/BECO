from typing import Dict

from torch import Tensor
from torch import nn
from torch.nn import functional as F

import modules
from utils.modules import init_weight


class Deeplabv3(nn.Module):
    """
    Returns full Deeplabv3
    This module has four components:

    self.backbone
    self.aspp
    self.preclassifier: an 3x3 conv for feature mixing, before final classification
    self.classifier: last 1x1 conv for output classification results

    Args:
        backbone: Dict, configs for backbone
        decoder: Dict, configs for decoder
    """
    def __init__(self, backbone: Dict, decoder: Dict) -> None:
        super(Deeplabv3, self).__init__()

        self.align_corners = decoder.settings.align_corners
        BN_op = getattr(nn, decoder.settings.norm_layer)
        channels = decoder.settings.channels
        self.backbone = modules.backbones.get_backbone(backbone)
        self.aspp = modules.decoders.get_decoder(decoder)
        self.pre_classifier = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            BN_op(channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(channels, decoder.settings.num_classes, 1, 1)

        init_weight(self.pre_classifier)
        init_weight(self.classifier)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        size = (x.shape[2], x.shape[3])
        out = self.backbone(x)

        output = self.aspp(out[4])
        output = self.pre_classifier(output)
        out['embeddings'] = output
        output = self.classifier(output)
        out['logits'] = F.interpolate(output, size=size, mode='bilinear', align_corners=self.align_corners)

        return out

    