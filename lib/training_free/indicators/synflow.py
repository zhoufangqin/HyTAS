import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import indicator
from ..p_utils import get_layer_metric_array, get_layer_metric_array_zico

@indicator('synflow', bn=True, mode='param') #bn=True/False does not make difference
def compute_synflow_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
    # # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients (but don't apply them)
    net.zero_grad()

    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).to(inputs.device)
    output, _ = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                # all of them using abs
                # https://github.com/lilujunai/DisWOT-CVPR2023/blob/gh-pages/zerocostproxy/SynFlow.py
                ##https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/synflow.py
                return torch.abs(layer.sampled_weight * layer.sampled_weight.grad)
                # return (layer.sampled_weight.grad, layer.sampled_weight)
                # return layer.sampled_weight * layer.sampled_weight.grad  #no abs according to the paper
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.samples['weight'].grad is not None:
                # all of them using abs
                # https://github.com/lilujunai/DisWOT-CVPR2023/blob/gh-pages/zerocostproxy/SynFlow.py
                ##https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/synflow.py
                return torch.abs(layer.samples['weight'] * layer.samples['weight'].grad)
                # return (layer.samples['weight'].grad, layer.samples['weight'])
                # return layer.samples['weight'] * layer.samples['weight'].grad #no abs according to the paper
            else:
                return torch.zeros_like(layer.samples['weight'])

    grads_abs = get_layer_metric_array_zico(net, synflow, mode)
    # # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs, net.sample_embed_dim[0]
