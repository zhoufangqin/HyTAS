## code from https://github.com/cmu-catalyst/GradSign/blob/main/naswot-code/gradsign.py#L65
## paper: https://arxiv.org/pdf/2110.08616.pdf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from . import indicator
from ..p_utils import get_layer_metric_array, get_flattened_metric


@indicator('gradsign', bn=True, mode='param')
def compute_grad_norm_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):

    def gradsign(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return layer.sampled_weight.grad.data.detach().cpu().numpy()
            else:
                return torch.zeros_like(layer.sampled_weight).data.detach().cpu().numpy()
        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.samples['weight'].grad is not None:
                return layer.samples['weight'].grad.data.detach().cpu().numpy()
            else:
                return torch.zeros_like(layer.samples['weight']).data.detach().cpu().numpy()

    N = inputs.shape[0]
    batch_grad = []
    for i in range(N):
        net.zero_grad()
        outputs, _ = net.forward(inputs[[i]])
        loss = loss_fn(outputs, targets[[i]])
        loss.backward()
        flattened_grad = get_flattened_metric(net, gradsign, mode)
        batch_grad.append(flattened_grad)

    batch_grad = np.stack(batch_grad)
    direction_code = np.sign(batch_grad)
    direction_code = abs(direction_code.sum(axis=0))
    score = np.nansum(direction_code)
    return np.log(score)
