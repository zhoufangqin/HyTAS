## using the mean and variance of the activation instead of gradients or weights as zico

import torch
import torch.nn as nn
import torch.nn.functional as F

import types

from . import indicator
from ..p_utils import get_layer_metric_array, reshape_elements
import numpy as np


def fisher_forward_conv2d(self, x):
    x = x.unsqueeze(2).permute(0, 3, 2, 1)
    x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size,
                 padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1, 2)
    # intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act

def fisher_forward_linear(self, x):
    x = F.linear(x, self.sampled_weight, self.sampled_bias)
    self.act = self.dummy(x)
    return self.act
def fisher_forward_linear_samples(self, x):
    x = F.linear(x, self.samples['weight'], self.samples['bias'])
    self.act = self.dummy(x)
    return self.act

@indicator('zico_act', bn=True, mode='channel')
def compute_zico_act_per_weight(net, inputs, targets, loss_fn, mode, split_data=1, pretrained_model=None):
    device = inputs.device

    if mode == 'param':
        raise ValueError('Fisher pruning does not support parameter pruning.')

    net.train()
    all_hooks = []
    for layer in net.modules():
        # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        if isinstance(layer, nn.Linear) or layer._get_name() == 'PatchembedSuper':
            # variables/op needed for fisher computation
            layer.fisher_per_sample = None
            layer.act = 0.
            layer.dummy = nn.Identity()

            # # replace forward method of conv/linear
            # if isinstance(layer, nn.Conv2d) and layer._get_name() == 'PatchembedSuper':
            if layer._get_name() == 'PatchembedSuper':
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Linear) and layer.samples:
                layer.forward = types.MethodType(fisher_forward_linear_samples, layer)

            # function to call during backward pass (hooked on identity op at output of layer)
            def hook_factory(layer):
                def hook(module, grad_input, grad_output):
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        # g_nk = torch.sum((act * grad), list(range(2,
                        #                                           len(act.shape))))  # sum up the last dimension (the length of the vectorized feature map)
                        g_nk = act * grad
                        g_nk = g_nk.reshape(g_nk.size(0), -1)
                    else:  # torch.Size([15, 15]) torch.Size([15, 15])
                        g_nk = act * grad
                    # del_k = g_nk.pow(2).mean(0).mul(0.5) #take mean over the number of samples
                    del_k = g_nk.pow(2)
                    if layer.fisher_per_sample is None:
                        layer.fisher_per_sample = del_k
                    else:
                        layer.fisher_per_sample += del_k
                    del layer.act  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555

                return hook

            # register backward hook on identity fcn to compute fisher info
            layer.dummy.register_backward_hook(hook_factory(layer))

    # retrieve fisher info
    def zico_act(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.fisher_per_sample is not None:
                # return layer.fisher_per_sample
                return layer.fisher_per_sample.detach()
            else:
                return torch.zeros(layer.sampled_weight.shape[0])  # size=ch
        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.fisher_per_sample is not None:
                # return layer.fisher_per_sample
                return layer.fisher_per_sample.detach()
            else:
                return torch.zeros(layer.samples['weight'].shape[0])  # size=ch
    # inputs = torch.randn([16, 200, 49 * 3]).cuda()
    N = inputs.shape[0]
    split_data = 1
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        net.zero_grad()
        outputs, loss_align = net(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    all_acts = get_layer_metric_array(net, zico_act, mode)

    return all_acts, net.sample_embed_dim[0]