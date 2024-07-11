import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import indicator
from ..p_utils import get_layer_metric_array, get_layer_metric_array_zico

@indicator('grad_entropy', bn=True, mode='param')
def compute_grad_entropy_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
    # convert params to their abs. Keep sign for converting it back.
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
    # inputs = torch.randn([16, 200, 49]).cuda()
    # inputs = torch.zeros([16, 200, 49]).cuda()
    N = inputs.shape[0]
    split_data = 1
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs, loss_align, attns = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def grad_entropy(layer):

        ## return both gradients and weights
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return [layer.sampled_weight, layer.sampled_weight.grad]
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.samples['weight'].grad is not None:
                return [layer.samples['weight'], layer.samples['weight'].grad]
            else:
                return torch.zeros_like(layer.samples['weight'])

    grads = get_layer_metric_array_zico(net, grad_entropy, mode)

    nonlinearize(net, signs)

    return grads, net.sample_embed_dim[0]
