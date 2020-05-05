"""modules used by CORnet models

CORnet by DiCarlo lab
https://github.com/dicarlolab/CORnet
adapted under GNU Public License
https://github.com/dicarlolab/CORnet/blob/master/LICENSE
"""
from torch import nn


class Flatten(nn.Module):
    """Helper module for flattening input tensor to 1-D for the use in Linear modules"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """Helper module that stores the current tensor. Useful for accessing by name"""
    def forward(self, x):
        return x
