# LS-CNN
PyTorch implementation of LS-CNN architecture from the paper

 - [LS-CNN: Characterizing Local Patches at Multiple Scales for Face Recognition](https://ieeexplore.ieee.org/abstract/document/8865656) by Qiangchang Wang and Guodong Guo


## About LS-CNN
Since similar discriminative face regions may  occur at different scales, a new backbone network HSNet which extracts multi-scale features has been proposed in the paper.  HSNet excracts multi-scale features from two harmonious perspectives.

 - Different kernels in a single layer, concatenation of features from different layers.
 - To identify local patches from which facial features can be extracted, new spatial attention is used in the paper.
 
	 
 - Also in a CNN, channels in high-level layers represent high-level representations. So, a channel attention is also used in the paper to emphasize important channels and suppress less informative ones automatically.
 -   This spatial and channel attention is called **Dual Face Attention (DFA)**
##
The goal of this package is to provide a nice and simple object-oriented implementation of the architecture. The individual submodules are cleanly separated into self-contained blocks, that come with documentation and typings, and that are therefore easy to import and reuse.
## Requirements and installation
This project is based on Python 3.6+ and PyTorch 1.7.0+

Within the correct environment, install the package from the repository:
```bash
pip install git+https://github.com/Ksuryateja/LS-CNN	
```
## Usage
Either load the predefined network from `model.py`
```Python
from model import LSCNN

net = LSCNN()
```
Or use the modules for custom architecture:
```Python
from torch.nn import Sequential
from LSCNN import TransitionBlock, DenseBlock

class Module(Sequential):
    def __init__(self, in_channels, out_channels):
        super(Module, self).__init__()

        self.dense = DenseBlock(in_channels, growth_rate=4, num_layers=2,
                                concat_input=True, dense_layer_params={'dropout': 0.2})

        self.transition = TransitionBlock(self.dense.out_channels, out_channels)

net = Module(10, 4)
```
   

    Module((dense): DenseBlock(60, 1*30+60=90)(
    (DenseBlock_0): Sequential(
      (DFA_0): DFA(60, 16)(
        (LANet): LALayer(
          (spatial_atten): Sequential(
            (0): Conv2d(60, 3, kernel_size=(1, 1), stride=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
            (3): Sigmoid()
          )
        )
        (SENet): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc1): Linear(in_features=60, out_features=3, bias=False)
          (relu): ReLU(inplace=True)
          (fc2): Linear(in_features=3, out_features=60, bias=False)
          (sigmoid): Sigmoid()
        )
      )
      (DenseLayer_0): DenseLayer(60, 30)(
        (branch1_1x1): BasicConv2d(
          (conv): Conv2d(60, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(120, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2_1x1): BasicConv2d(
          (conv): Conv2d(60, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(120, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3_1x1): BasicConv2d(
          (conv): Conv2d(60, 30, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch1_3x3): BasicConv2d(
          (conv): Conv2d(120, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch1_3x3_2): BasicConv2d(
          (conv): Conv2d(30, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2_3x3): BasicConv2d(
          (conv): Conv2d(120, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (output): BasicConv2d(
          (conv): Conv2d(90, 30, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
     )
    )
    (transition): TransitionBlock()(
    (transition_block): Sequential(
      (DFA): DFA(90, 16)(
        (LANet): LALayer(
          (spatial_atten): Sequential(
            (0): Conv2d(90, 5, kernel_size=(1, 1), stride=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(5, 1, kernel_size=(1, 1), stride=(1, 1))
            (3): Sigmoid()
          )
        )
        (SENet): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc1): Linear(in_features=90, out_features=5, bias=False)
          (relu): ReLU(inplace=True)
          (fc2): Linear(in_features=5, out_features=90, bias=False)
          (sigmoid): Sigmoid()
        )
      )
      (Transition layer): TransitionLayer(90, 45)(
        (branch1_1x1): BasicConv2d(
          (conv): Conv2d(90, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(45, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2_1x1): BasicConv2d(
          (conv): Conv2d(90, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(45, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3_1x1): BasicConv2d(
          (conv): Conv2d(90, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(45, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch1_3x3): BasicConv2d(
          (conv): Conv2d(45, 45, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(45, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch1_3x3_2): BasicConv2d(
          (conv): Conv2d(45, 45, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (bn): BatchNorm2d(45, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2_3x3): BasicConv2d(
          (conv): Conv2d(45, 45, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (bn): BatchNorm2d(45, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3_3x3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (output): BasicConv2d(
          (conv): Conv2d(135, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(45, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
     ) 
    )
    )

