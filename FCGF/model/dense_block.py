import torch.nn as nn
import torch
from model.common import get_norm

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None,
                 bn_momentum=0.1,
                 D=3):
        super(BasicBlockBase, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False, dimension=D
        )

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False, dimension=D
        )
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=out_channels * 3,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False, dimension=D
        )
        self.conv4 = ME.MinkowskiConvolution(
            in_channels=out_channels * 4,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False, dimension=D
        )
        self.norm = get_norm(self.NORM_TYPE,self.conv1.out_channels+self.conv2.out_channels+self.conv3.out_channels+self.conv4.out_channels, bn_momentum=bn_momentum, D=D)
    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(ME.cat([x, x1]))
        x3 = self.conv3(ME.cat([x, x1, x2]))
        x4 = self.conv4(ME.cat([x, x1, x2, x3]))
        out = ME.cat([x1, x2, x3, x4])
        out=self.norm(out)
        return out


class BasicBlockBN(BasicBlockBase):
    NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = 'IN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
    if norm_type == 'BN':
        return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    elif norm_type == 'IN':
        return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')
