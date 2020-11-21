import torch.nn as nn


class LALayer(nn.Module):
    def __init__(
        self, 
        in_channel: int,
        reduction: int = 16):
        super(LALayer, self).__init__()
        self.spatial_atten = nn.Sequential(
                nn.Conv2d(in_channel, in_channel // reduction, kernel_size = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel // reduction, 1, kernel_size = 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        
        y = self.spatial_atten(x)
        return x * y
