from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .la_module import LALayer
from .se_module import SELayer
from ..utils import RichRepr

class DFA(RichRepr, nn.Module):
    def __init__(
        self, 
        in_channels: int,
        reduction: Optional[int] = 16) -> None:
        self.in_channels = in_channels
        self.reduction = reduction

        super(DFA, self).__init__()
        self.LANet = LALayer(in_channels, reduction)
        self.SENet = SELayer(in_channels, reduction)
        
    def __repr__(self):
        return super(DFA, self).__repr__(self.in_channels, self.reduction)

    def forward(self, x: Tensor) -> Tensor:
        output = self.SENet(self.LANet(x))
    
        return output