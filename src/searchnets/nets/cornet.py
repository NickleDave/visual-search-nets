"""CORnet by DiCarlo lab
https://github.com/dicarlolab/CORnet
adapted under GNU Public License
https://github.com/dicarlolab/CORnet/blob/master/LICENSE
"""
from collections import OrderedDict

import torch
from torch import nn
import torch.utils.model_zoo


HASH = '5c427c9c'


class Flatten(nn.Module):
    """Helper module for flattening input tensor to 1-D for the use in Linear modules"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """Helper module that stores the current tensor. Useful for accessing by name"""
    def forward(self, x):
        return x


class CORblock_Z(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


def CORnet_Z(num_classes=1000, apply_sigmoid=False):
    """returns CORnet_Z model"""
    model_list = [
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, num_classes)),
            ('output', Identity())
        ])))
    ]
    if apply_sigmoid:
        model_list.append(
            ('sigmoid', nn.Sigmoid())
        )
    model = nn.Sequential(OrderedDict(model_list))

    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


def build(pretrained=False, map_location=None, **kwargs):
    model = CORnet_Z(**kwargs)
    if pretrained:
        model_hash = HASH
        model_letter = 'z'
        url = f'https://s3.amazonaws.com/cornet-models/cornet_{model_letter.lower()}-{model_hash}.pth'
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=map_location)
        # remove the 'module.' from keys in state_dict, since we don't have the model wrapped in nn.DataParallel yet
        ckpt_data['state_dict'] = {k.replace('module.', ''): v for k, v in ckpt_data['state_dict'].items()}
        model.load_state_dict(ckpt_data['state_dict'])
    return model


def reinit(model, num_classes=2):
    """re-initialize linear output layer"""
    model.decoder.linear = nn.Linear(in_features=512, out_features=num_classes)
    return model
