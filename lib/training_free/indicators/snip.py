import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from . import indicator
from ..p_utils import get_layer_metric_array


def snip_forward_conv2d(self, x):
        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        return F.conv2d(x, self.sampled_weight * self.weight_mask, self.sampled_bias,
                        stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        
def snip_forward_linear(self, x):
        return F.linear(x, self.samples['weight'] * self.weight_mask, self.samples['bias'])


def snip_forward_linear_(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

@indicator('snip', bn=True, mode='param')
def compute_snip_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
    for layer in net.modules():
        if layer._get_name() == 'PatchembedSuper':
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.sampled_weight))
            layer.sampled_weight = layer.sampled_weight.detach()
        if isinstance(layer, nn.Linear) and layer.samples: #is 10 the number of classes?
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.samples['weight']))
            layer.samples['weight'] = layer.samples['weight'].detach()
        # Override the forward methods:
        if layer._get_name() == 'PatchembedSuper':
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear) and layer.samples:
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    # inputs = torch.randn([16,200,49*3]).cuda()
    # inputs = torch.zeros([16, 200, 49]).cuda()
    N = inputs.shape[0]
    split_data = 1
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        outputs, loss_align = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.weight_mask.grad is not None:
                return torch.abs(layer.weight_mask.grad)
            else:
                return torch.zeros_like(layer.weight)
        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.weight_mask.grad is not None:
                return torch.abs(layer.weight_mask.grad)

            else:
                return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, snip, mode)
    return grads_abs, (net.sample_embed_dim[0], net.sample_num_heads)
