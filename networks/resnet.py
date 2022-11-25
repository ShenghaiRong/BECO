"""
Adapted from ResNet Pytorch implementation for Deeplab
"""
from typing import Dict, Type, Callable, Union, List, Optional, Tuple

import torch
import torch.nn as nn
from collections import OrderedDict

from utils.modules import init_weight


def conv3x3(
        in_planes: int, out_planes: int,
        stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class HeadLayer(nn.Module):

    def __init__(self, BN_ops=nn.BatchNorm2d):
        super(HeadLayer, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN_ops(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        return out


class HeadLayer_C(nn.Module):
    """ResNet-C variety"""

    def __init__(self, BN_ops=nn.BatchNorm2d):
        super(HeadLayer_C, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BN_ops(32)
        self.relu = nn.ReLU(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BN_ops(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BN_ops(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        return out
    

class BasicBlock(nn.Module):
    """
    BasicBlock only supports groups=1 and base_width=64
    """
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out
        

class Bottleneck(nn.Module):
    """
    Implemetation of ResNet-B, which move stride from 1x1conv to 3x3conv
    Refer to 1812.01187 for details

    Use groups and base_width to construct ResNeXt blocks
    """
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:

        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Modified ResNet w/o FC layer for deeplab, added some custom features
    """

    def __init__(
        self,
        variety,
        block: Union[BasicBlock, Bottleneck],
        layers: List[int],
        out_indices: List[int],
        strides: List[int],
        dilations: List[int],
        contract_dilation: bool,
        multi_grid: List[int],
        norm_layer: Callable[..., nn.Module],
        groups: int=1,
        width_per_group: int=64,
    ) -> None:

        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.out_indices = out_indices
        self.inplanes = 64

        if variety == "resnet-B":
            self.type_C = False
            self.type_D = False
        elif variety == "resnet-C":
            self.type_C = True
            self.type_D = False
        elif variety == "resnet-D":
            self.type_C = True
            self.type_D = True
        
        # First conv layer
        if self.type_C:
            self.stem_layer = HeadLayer_C(norm_layer)
        else:
            self.stem_layer = HeadLayer(norm_layer)
        # Residual blocks
        # multi-grid is only applied to the last block
        self.layer1 = self._make_layer(
            block, 64, layers[0],
            stride=strides[0],
            dilation=dilations[0],
            contract_dilation=contract_dilation,
            multi_grid=None
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=strides[1],
            dilation=dilations[1],
            contract_dilation=contract_dilation,
            multi_grid=None
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=strides[2],
            dilation=dilations[2],
            contract_dilation=contract_dilation,
            multi_grid=None
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=strides[3],
            dilation=dilations[3],
            contract_dilation=contract_dilation,
            multi_grid=multi_grid
        )

        init_weight(self)
        # Zero-initialize the last BN in each residual branch, so that the 
        # residual branch starts with zeros, and each residual block behaves 
        # like an identity. This improves the model by 0.2~0.3% 
        # according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block: Type[Union[BasicBlock, Bottleneck]], 
        planes: int, blocks: int, stride: int = 1, dilation: int = 1,
        contract_dilation: bool=False, multi_grid: List[int]=None
    ) -> nn.Sequential:

        layers = []
        norm_layer = self.norm_layer
        # stride should be 1 if dilation is applied
        if dilation != 1:
            stride = 1
        # multi_grid and contract_dilation cannot be applied together
        if multi_grid is not None:
            first_dilation = multi_grid[0] * dilation
        elif contract_dilation and dilation > 1:
            first_dilation = dilation // 2
        else:
            first_dilation = dilation

        # downsample is required for 1st bottleneck of a residual group, include:
        # 1. Using stride
        # 2. stride = 1 but dilation is used
        if stride != 1 or self.inplanes != planes * block.expansion:
            # ResNet-D implementation
            if self.type_D and stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(2, stride=2),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                )
            # AvgPool2d is not used when dilation != 1
            # add an placeholder to make parameters loading correctly
            elif self.type_D and dilation != 1:
                downsample = nn.Sequential(
                    nn.Identity(),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                )
            # ResNet-B implementation
            # Note that layer1 of ResNet is both stride = 1 and dilation = 1 and goes here
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
        else:
            downsample = None

        # First conv layer, stride is only applied here
        layers.append(block(
            self.inplanes, planes, stride, self.groups,
            self.base_width, first_dilation, downsample, norm_layer
        ))
        self.inplanes = planes * block.expansion
        # Following conv lyaers
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width,
                dilation=dilation if multi_grid is None else multi_grid[i] * dilation,
                norm_layer=norm_layer
            ))
        return nn.Sequential(*layers)

    def forward(self, x) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        out = dict()
        x = self.stem_layer(x)
        for i in range(1, 5):
            layer = getattr(self, f"layer{i}")
            x = layer(x)
            if i in self.out_indices:
                out[i] = x
        return out


def get_convnet(
    pretrain: str,
    depth: int,
    variety: str,
    out_indices: List[int],
    output_stride: int,
    norm_layer: str,
    multi_grid: bool,
    contract_dilation: bool,
) -> ResNet:

    # multi_grid and contract_dilation cannot be applied together
    assert (not multi_grid or not contract_dilation)
    _multi_grid = [1, 2, 4] if multi_grid else None
    _norm_layer = getattr(nn, norm_layer)

    if depth < 50:
        _block = BasicBlock
    else:
        _block = Bottleneck

    layers_table = {
        18 : [2, 2, 2, 2],
        32 : [3, 4, 6, 3],
        50 : [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 4, 36, 3],
    }

    strides_table = {
        8 : [1, 2, 1, 1],
        16: [1, 2 ,2, 1],
    }

    dilations_table = {
        8 : [1, 1, 2, 4],
        16: [1, 1, 1, 2],
    }

    convnet = ResNet(
        variety=variety, block=_block, layers=layers_table[depth], out_indices=out_indices,
        strides=strides_table[output_stride], dilations=dilations_table[output_stride], contract_dilation=contract_dilation, multi_grid=_multi_grid, norm_layer=_norm_layer
    )
    
    if pretrain:
        state_dict = torch.load(pretrain, map_location='cpu')
        state_dict = rename_mmcv_state_dict_keys(state_dict)
        # Remove last FC layer
        if 'fc.weight' in state_dict.keys():
            del state_dict['fc.weight'], state_dict['fc.bias']
        convnet.load_state_dict(state_dict)
        del state_dict

    return convnet


def rename_mmcv_state_dict_keys(source_state_dict):
    source_state_dict = source_state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in source_state_dict.items():
        # Remove 'num_batches_tracked'
        if 'num_batches_tracked' in k:
            continue

        if 'backbone' in k:
            # Remove 'backbone.'
            k = k[9:]
            # Process stem layer
            if 'layer' not in k:
                if 'stem' in k:
                    num = str(int(k[5]) + 1)
                    if 'conv' in k:
                        k = k[7:11] + num + k[11:]
                    elif 'bn' in k:
                        k = k[7:9] + num + k[9:]
                k = 'stem_layer.' + k

            new_state_dict[k] = v
        # another case in mmcv ckpt
        elif 'head' in k:
            k = k[5:]
            new_state_dict[k] = v
        else:
            print(f"warning: {k} not processed")
    return new_state_dict
    

