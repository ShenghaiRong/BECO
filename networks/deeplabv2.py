from typing import Dict

from torch import Tensor
from torch import nn
import torch
from torch.nn import functional as F

import modules
from utils.modules import init_weight


class Deeplabv2(nn.Module):
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
        super(Deeplabv2, self).__init__()

        self.align_corners = decoder.settings.align_corners
        BN_op = getattr(nn, decoder.settings.norm_layer)
        channels = decoder.settings.channels
        self.msc = decoder.settings.get('msc', False)
        self.scales = decoder.settings.get('scales', [0.5, 0,75])
        self.backbone = modules.backbones.get_backbone(backbone)
        self.aspp = ASPP(decoder.settings.in_channels,
                         decoder.settings.num_classes,
                         decoder.settings.atrous_rates)
        #self.aspp = modules.decoders.get_decoder(decoder)
        #self.pre_classifier = nn.Sequential(
        #    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        #    BN_op(channels),
        #    nn.ReLU(inplace=True),
        #)
        #self.classifier = nn.Conv2d(channels, decoder.settings.num_classes, 1, 1)

        #init_weight(self.pre_classifier)
        #init_weight(self.classifier)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        size = (x.shape[2], x.shape[3])
        out = self.backbone(x)
        output = self.aspp(out[4])

        if self.msc:
            logits = [output]
            h, w = logits[0].shape[2:]        
            for i in self.scales:
                x_i = F.interpolate(x, scale_factor=i, mode='bilinear',
                                    align_corners=self.align_corners)
                l_i = self._base_forward(x_i)
                l_i = F.interpolate(l_i, size=(h,w), mode='bilinear',
                                    align_corners=self.align_corners)
                logits.append(l_i)
            l_max = torch.max(torch.stack(logits), dim=0)[0]
            logits.append(l_max)
            output = torch.sum(logits)
        out['pre_logits'] = output
        out['logits'] = F.interpolate(output, size=size, mode='bilinear', 
                                      align_corners=self.align_corners)

        return out

    def _base_forward(self, x):
        out = self.backbone(x)
        out = self.aspp(out[4])
        return out

    def _pyramid(self, x):
        #h, w = x.shape[2:]
        x_scales = []
        logits = [self._base_forward(x)]
        h, w = logits[0].shape[2:]
        for i in self.scales:
            x_i = F.interpolate(x, scale_factor=i, mode='bilinear', align_corners=False)
            l_i = self._base_forward(x_i)
            l_i = F.interpolate(l_i, size=(h, w), mode='bilinear', align_corners=False)
            logits.append(l_i)
        '''
        logits = [self.base(x)]
        for l in x_scales:
            _temp = F.interpolate(l, size=(h, w), mode='bilinear', align_corners=False)
            logits.append(self.base(_temp))
        '''
        x_max = torch.max(torch.stack(logits), dim=0)[0]

        if self.training:
            return logits + [x_max]
        else:
            return x_max


class ASPP(nn.Module):
    def __init__(self, in_channels, channels, atrous_rates):
        super(ASPP, self).__init__()
        for i, rate in enumerate(atrous_rates):
            self.add_module("c%d"%(i), nn.Conv2d(in_channels, channels, 3, 1, 
                                        padding=rate, dilation=rate, bias=True))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])