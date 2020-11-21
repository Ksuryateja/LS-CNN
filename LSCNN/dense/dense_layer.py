from typing import Callable, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.tensor import Tensor
from ..utils import RichRepr

class DenseLayer(RichRepr, nn.Module):
    r"""
    Inception Module with dimension reduction as described in [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)
    Consists of:
    - 5x5 is replaced with 3x3
    - Pooling branch is dropped
    - 3x3 Convolution
    - 1x1 Convolution
    """

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            growth_rate: int,
            conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super(DenseLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if conv_block == None:
            conv_block = BasicConv2d

        self.branch1_1x1 = conv_block(in_channels, 4 * growth_rate, kernel_size = 1)
        self.branch2_1x1 = conv_block(in_channels, 4 * growth_rate, kernel_size = 1)
        self.branch3_1x1 = conv_block(in_channels, growth_rate, kernel_size = 1)

        self.branch1_3x3 = conv_block(4 * growth_rate, growth_rate, kernel_size = 3, padding = 1)
        self.branch1_3x3_2 = conv_block(growth_rate, growth_rate, kernel_size = 3, padding = 1)

        self.branch2_3x3 = conv_block(4 * growth_rate, growth_rate, kernel_size = 3, padding = 1)
        
        self.output = conv_block(3 * growth_rate, out_channels, kernel_size = 1)

    def _forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1_3x3_2(self.branch1_3x3(self.branch1_1x1(x)))
        branch2 = self.branch2_3x3(self.branch2_1x1(x))
        branch3 = self.branch3_1x1(x)

        #print(f'branch1:{branch1.size()}, branch2:{branch2.size()}, branch3:{branch3.size()}')
        output = self.output(torch.cat([branch1, branch2, branch3], dim = 1))
        
        return output
        
    def __repr__(self):
        return super(DenseLayer, self).__repr__(self.in_channels, self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        output = self._forward(x)

        return output

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)