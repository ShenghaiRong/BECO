from typing import List

import torch
from torch import Tensor, norm
from torch import nn
from torch.nn import functional as F

from utils.modules import init_weight


class ASPP(nn.Module):
    """
    NOTE: The projector kernel_size is 1 by default, but some place will use 3
    """
    def __init__(
        self, in_channels: int, channels: int, atrous_rates: List[int],
        norm_layer: str='SyncBatchNorm', dropout_ratio: float=0.1,
        align_corners: bool=False, **kwargs
    ):
        super(ASPP, self).__init__()
        self.BN_op = getattr(torch.nn, norm_layer)

        # All Atrous convs
        modules = []
        # the 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            self.BN_op(channels),
            nn.ReLU(inplace=True)
        ))
        # the 3x3 atrous convs
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, channels, rate, self.BN_op))
        # the image pooling
        modules.append(ASPPPooling(in_channels, channels, self.BN_op, align_corners))

        self.convs = nn.ModuleList(modules)
        self.projector = nn.Sequential(
            nn.Conv2d(len(self.convs) * channels, channels, 1, bias=False),
            self.BN_op(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio)
        )

        init_weight(self)
    
    def forward(self, x: Tensor) -> Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.projector(res)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, BN_op, align_corners):
        super(ASPPPooling, self).__init__()
        self.align_corners = align_corners
        self.convs = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BN_op(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        x = self.convs(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=self.align_corners)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, BN_op):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            BN_op(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)