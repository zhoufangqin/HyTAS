import numpy as np
import torch
import torch.nn as nn

from . import indicator
from ..p_utils import get_layer_metric_array


@indicator('cond_num', bn=True, mode='param')
def compute_cond_num_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
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
                U, S, V = torch.svd(layer.sampled_weight.squeeze())
                # print(torch.sum(torch.abs(layer.sampled_weight * layer.sampled_weight.grad)).item(), torch.log(S[0]/S[-1]).item())
                return torch.sum(torch.abs(layer.sampled_weight * layer.sampled_weight.grad)) / (
                    torch.log(S[0] / S[-1] + 1e-8)) #almost the same as snip
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.samples['weight'].grad is not None:
                U, S, V = torch.svd(layer.samples['weight'])
                # print(torch.sum(torch.abs(layer.samples['weight'] * layer.samples['weight'].grad)).item(), torch.log(S[0] / S[-1]).item())
                return torch.sum(torch.abs(layer.samples['weight'] * layer.samples['weight'].grad)) / (
                    torch.log(S[0] / S[-1] + 1e-8))
            else:
                return torch.zeros_like(layer.samples['weight'])

    grads_norm = get_layer_metric_array(net, grad_norm, mode)

    return grads_norm, net.sample_embed_dim[0]
