import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import indicator
from ..p_utils import get_layer_metric_array, get_layer_metric_array_zico

@indicator('logsynflow', bn=True, mode='param')
def compute_logsynflow_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
    net.train()
    # # Disable batch norm
    # for layer in net.modules():
    #     if isinstance(layer, _BatchNorm):
    #         # TODO: this could be done with forward hooks
    #         layer._old_forward = layer.forward
    #         layer.forward = types.MethodType(_no_op, layer)

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

    # # Compute gradients (but don't apply them)
    net.zero_grad()
    input_dim = list(inputs[0, :].shape)
    # print(input_dim) #(200, 7*7*3)
    inputs = torch.ones([1] + input_dim).to(inputs.device)
    # print(inputs.size()) #torch.Size([1, 200, 147])
    output, _ = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def logsynflow(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return torch.abs(torch.log(layer.sampled_weight.grad + 1) * layer.sampled_weight)
                # return (layer.sampled_weight.grad, layer.sampled_weight)
            else:
                return torch.zeros_like(layer.sampled_weight)

        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.samples['weight'].grad is not None:
                return torch.abs(torch.log(layer.samples['weight'].grad + 1) * layer.samples['weight'])
                # return (layer.samples['weight'].grad, layer.samples['weight'])
            else:
                return torch.zeros_like(layer.samples['weight'])

    grads_abs = get_layer_metric_array_zico(net, logsynflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs, net.sample_embed_dim[0]
