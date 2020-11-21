from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .transition_layer import TransitionLayer
from ..DFA import DFA
from ..utils import RichRepr

class TransitionBlock(RichRepr, nn.Module):
    def __init__(
        self, 
        in_channels: int,
        transition_layer_params: Optional[dict] = None) -> None:

        super(TransitionBlock, self).__init__()
        
        self.in_channels = in_channels
        

        if transition_layer_params is None:
            transition_layer_params = {}

        self.transition_block = nn.Sequential()
        self.transition_block.add_module(f'DFA', DFA(in_channels))
        self.transition_block.add_module(f'Transition layer', TransitionLayer(in_channels))  
    
    def forward(self, x: Tensor) -> Tensor:
        output = self.transition_block(x)
        return output

'''    def __repr__(self) -> str:
        concat_input = f'+{self.in_channels}' if self.concat_input else ''
        out_channels = f'{self.num_layers}*{self.growth_rate}{concat_input}={self.out_channels}'
        return super(TransitionBlock, self).__repr__(self.in_channels, out_channels)
'''