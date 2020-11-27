import torch
import torch.nn as nn

from torch import Tensor
import torch.utils.data as data
import torchvision.transforms as transforms


from typing import Optional, Union, Sequence  
from math import ceil

from LSCNN import DenseBlock, TransitionBlock


class LSCNN(nn.Sequential):
    def __init__(
        self,
        num_classes: int = 10559,
        growth_rate: int = 48,
        num_layers: Sequence[int] = (3, 3, 5)
        ):
        super().__init__()
        self.growth_rate = growth_rate
        D1_in = 2 * growth_rate
        T1_in = D1_in + num_layers[0] * growth_rate
        D2_in = int(ceil(T1_in * 0.5))
        T2_in = D2_in + num_layers[1] * growth_rate
        D3_in = int(ceil(T2_in * 0.5))
        classif_in = D3_in + num_layers[2] * growth_rate

        # Feature Block
        self.add_module('conv1', nn.Conv2d(3, growth_rate, kernel_size = 3, padding = 1))
        self.add_module('norm1', nn.BatchNorm2d(growth_rate)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        
        self.add_module('conv2', nn.Conv2d(growth_rate, growth_rate, kernel_size = 3, padding = 1))
        self.add_module('norm2', nn.BatchNorm2d(growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),

        self.add_module('maxPool1', nn.MaxPool2d(3, 2))
        self.add_module('conv3', nn.Conv2d(growth_rate, 2 * growth_rate, kernel_size = 3, padding = 1))
        self.add_module('norm3', nn.BatchNorm2d(2 * growth_rate)),
        self.add_module('relu3', nn.ReLU(inplace=True)),
        
        self.add_module('conv4', nn.Conv2d(2 * growth_rate, 2 * growth_rate, kernel_size = 3, padding = 1))
        self.add_module('norm4', nn.BatchNorm2d(2 * growth_rate)),
        self.add_module('relu4', nn.ReLU(inplace=True)),

        self.add_module('maxPool2', nn.MaxPool2d(3, 2))


        self.add_module('lcnn_D1', DenseBlock(in_channels = D1_in, growth_rate = growth_rate, num_layers = num_layers[0]))
        self.add_module('lscnn_T1', TransitionBlock(in_channels = T1_in))
        self.add_module('lscnn_D2', DenseBlock(in_channels = D2_in, growth_rate = growth_rate, num_layers = num_layers[1]))
        self.add_module('lscnn_T2', TransitionBlock(in_channels = T2_in ))
        self.add_module('lscnn_D3', DenseBlock(in_channels = D3_in, growth_rate = growth_rate, num_layers = num_layers[2]))
        # CLassification Block
        self.add_module('norm', nn.BatchNorm2d(num_features= classif_in))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('avgPool', nn.AvgPool2d(kernel_size = 7, stride = 1))
        self.add_module('flatten', nn.Flatten())
        self.add_module('fc', nn.Linear(D3_in + num_layers[2] * growth_rate, 512))
        self.add_module('classification', nn.Linear(512, num_classes))

        #self.add_module('softmax', nn.Softmax(dim = 1))

        def forward(self, x: Tensor) -> Tensor:
            return super().forward(x)
