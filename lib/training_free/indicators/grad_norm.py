import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from . import indicator
from ..p_utils import get_layer_metric_array


@indicator('grad_norm', bn=True, mode='param')
def compute_grad_norm_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
    print(inputs.shape)
    net.zero_grad()
    # inputs = torch.randn([16, 200, 49*3]).cuda()
    N = inputs.shape[0]
    split_data = 1
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs, loss_align = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    def grad_norm(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return layer.sampled_weight.grad.norm()
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.samples['weight'].grad is not None:
                return layer.samples['weight'].grad.norm()
            else:
                return torch.zeros_like(layer.samples['weight'])

    grads_norm = get_layer_metric_array(net, grad_norm, mode)

    return grads_norm, net.sample_embed_dim[0]
