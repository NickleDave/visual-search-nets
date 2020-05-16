"""CORnet by DiCarlo lab
https://github.com/dicarlolab/CORnet
adapted under GNU Public License
https://github.com/dicarlolab/CORnet/blob/master/LICENSE
"""
import sys

import torch
from torch import nn
import torch.utils.model_zoo

from . import cornet_z, cornet_rt, cornet_s
model_module_map = {
    module_name.split('.')[-1]: module_name
    for module_name in [cornet_z.__name__, cornet_rt.__name__, cornet_s.__name__]
}

VALID_MODEL_NAMES = frozenset(
    model_module_map.keys()
)


def build(model_name, pretrained=False, map_location=None, **kwargs):
    model_name = model_name.lower()
    if model_name not in VALID_MODEL_NAMES:
        raise ValueError(
            f'Model name not recognized: {model_name}.\n'
            f'Valid model names are: {VALID_MODEL_NAMES}'
        )
    module_name = model_module_map[model_name]
    model_module = sys.modules[module_name]
    model = model_module.MODEL(**kwargs)
    if pretrained:
        model_hash = model_module.HASH
        model_letter = model_module.MODEL_LETTER
        url = f'https://s3.amazonaws.com/cornet-models/cornet_{model_letter}-{model_hash}.pth'
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=map_location)
        # remove the 'module.' from keys in state_dict, since we don't have the model wrapped in nn.DataParallel yet
        ckpt_data['state_dict'] = {k.replace('module.', ''): v for k, v in ckpt_data['state_dict'].items()}
        model.load_state_dict(ckpt_data['state_dict'])
    return model


def reinit(model, model_name, num_classes=2):
    """re-initialize linear output layer"""
    model.decoder.linear = nn.Linear(in_features=512, out_features=num_classes)
    if model_name.lower() == 'cornet_z':
        # cornet z linear layer initialized this way
        nn.init.xavier_uniform_(model.decoder.linear.weight)
        if model.decoder.linear.bias is not None:
            nn.init.constant_(model.decoder.linear.bias, 0)
    elif model_name.lower() == 'cornet_s':
        # no unique initializiation for cornet s,
        # see comment in model's module
        pass
    else:
        raise ValueError(
            f'model name not recognized: {model_name}'
        )
    return model
