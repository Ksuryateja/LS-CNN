from typing import Optional

import torch
import torch.nn as nn

from .dense_layer import DenseLayer
from ..DFA import DFA
from ..utils import RichRepr

class DenseBlock(RichRepr, nn.Module):
    def __init__(
        self, 
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        concat_input: bool = True,
        dfa_layer_params: Optional[dict] = None,
        dense_layer_params: Optional[dict] = None) -> None:

        super(DenseBlock, self).__init__()
        
        self.concat_input = concat_input
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.out_channels = growth_rate * num_layers

        if self.concat_input:
            self.out_channels += self.in_channels
        
        if dfa_layer_params is None:
            dfa_layer_params = {}
        
        if dense_layer_params is None:
            dense_layer_params = {}
        
        
        for i in range(num_layers):
            dense_block = nn.Sequential()
            dense_block.add_module(f'DFA_{i}', DFA(in_channels = in_channels + i * growth_rate, **dfa_layer_params))
            dense_block.add_module(f'DenseLayer_{i}',
                DenseLayer(in_channels = in_channels + i * growth_rate, out_channels = growth_rate, growth_rate = growth_rate, **dense_layer_params))
            self.add_module(f'DenseBlock_{i}', dense_block)    
            '''self.add_module(
                f'DFA_{i}', DFA(in_channels = in_channels + i * growth_rate, **dfa_layer_params)
            )
            self.add_module(
                f'DenseLayer_{i}',
                DenseLayer(in_channels = in_channels + i * growth_rate, out_channels = growth_rate, growth_rate = growth_rate, **dense_layer_params)
            ) '''
    
    def forward(self, block_input):
        layer_input = block_input
        layer_output = block_input.new_empty(0)

        all_outputs = [block_input] if self.concat_input else []
        
        for _,layer in self._modules.items():
            layer_input = torch.cat([layer_input, layer_output], dim = 1)
            layer_output = layer(layer_input)
            all_outputs.append(layer_output)
        
        return torch.cat(all_outputs, dim =1)

    def __repr__(self) -> str:
        concat_input = f'+{self.in_channels}' if self.concat_input else ''
        out_channels = f'{self.num_layers}*{self.growth_rate}{concat_input}={self.out_channels}'
        return super(DenseBlock, self).__repr__(self.in_channels, out_channels)
