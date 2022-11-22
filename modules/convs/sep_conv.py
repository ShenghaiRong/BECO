from torch import nn

class DepthwiseSeparableConv(nn.Module):
    """ Depthwise Separable Convolution"""
    def __init__(
            self, BN_op: str, in_channels, out_channels,
            kernel_size, stride=1, padding=0, dilation=1
        ):
        super(DepthwiseSeparableConv, self).__init__()

        BN_op = getattr(nn, BN_op)
        self.convs = nn.Sequential(
            # depthwise conv
            nn.Conv2d(
                in_channels, in_channels, kernel_size, stride, padding,
                dilation=dilation, bias=False, groups=in_channels
            ),
            BN_op(in_channels),
            nn.ReLU(inplace=True),
            # pointwise conv
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False
            ),
            BN_op(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)