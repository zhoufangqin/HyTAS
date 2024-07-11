import torch
import torch.nn as nn
import torch.nn.functional as F

import types

from . import indicator
from ..p_utils import get_layer_metric_array, reshape_elements


def fisher_forward_conv2d(self, x):
    x = x.unsqueeze(2).permute(0, 3, 2, 1)
    x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size,
                 padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1, 2)
    # intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    # self.act = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    return self.act

def fisher_forward_linear_samples(self, x):
    x = F.linear(x, self.samples['weight'], self.samples['bias'])
    self.act = self.dummy(x)
    # self.act = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    return self.act

## for pretrained model
def fisher_forward_linear_pre(self, x):
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    # self.act = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    return self.act

@indicator('mi', bn=True, mode='channel')
def compute_mi_per_weight(net, inputs, targets, loss_fn, mode, split_data=1, pretrained_model=None):
    device = inputs.device

    if mode == 'param':
        raise ValueError('Fisher pruning does not support parameter pruning.')

    def compute_mi(network, pretrained=False):
        network.train()
        all_hooks = []
        for layer in network.modules():
            # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) or layer._get_name() == 'PatchembedSuper':
                # variables/op needed for fisher computation
                layer.fisher = None
                layer.act = 0.
                layer.dummy = nn.Identity()

                # # replace forward method of conv/linear
                # if isinstance(layer, nn.Conv2d) and layer._get_name() == 'PatchembedSuper':
                if layer._get_name() == 'PatchembedSuper':
                    layer.forward = types.MethodType(fisher_forward_conv2d, layer)
                if isinstance(layer, nn.Linear):
                    if pretrained:
                        layer.forward = types.MethodType(fisher_forward_linear_pre, layer)
                    else:
                        if layer.samples:
                            layer.forward = types.MethodType(fisher_forward_linear_samples, layer)

                # function to call during backward pass (hooked on identity op at output of layer)
                def hook_factory(layer):
                    def hook(module, grad_input, grad_output):
                        # print(grad_input[0].shape) #torch.Size([15*145, emb*])
                        act = layer.act.detach()
                        # grad = grad_output[0].detach()
                        # print(act.shape, grad.shape) #torch.Size([15, 145, emb*]) torch.Size([15, 145, emb*]) or 144
                        if len(act.shape) > 2:
                            # g_nk = torch.sum((act * grad), list(range(2, len(act.shape)))) #sum up the last dimension (the length of the vectorized feature map)
                            g_nk = act
                        else: #torch.Size([15, 15]) torch.Size([15, 15])
                            # g_nk = act * grad
                            g_nk = act
                        # print(g_nk.shape)
                        # del_k = F.normalize(g_nk.pow(2).mean(1).view(g_nk.size(0), -1))
                        del_k = F.normalize(g_nk.pow(2).mean(-1).view(g_nk.size(0), -1))
                        # del_k = g_nk.pow(2).mean(0).mul(0.5) #take mean over the number of samples
                        # print(g_nk.shape) #torch.Size([15, 15]), torch.Size([15, 145]) ..., torch.Size([15, 144])
                        # print(del_k.shape) #torch.Size([15]), torch.Size([145]) ..., torch.Size([144])
                        if layer.fisher is None:
                            layer.fisher = del_k
                        else:
                            layer.fisher += del_k
                        del layer.act  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555

                    return hook

                layer.dummy.register_backward_hook(hook_factory(layer))

        # retrieve fisher info
        def mi(layer):
            if pretrained:
                if layer.fisher is not None:
                    return torch.abs(layer.fisher.detach())
                else:
                    return torch.zeros(layer.weight.shape[0])
            else:
                if layer._get_name() == 'PatchembedSuper':
                    if layer.fisher is not None:
                        return torch.abs(layer.fisher.detach())
                    else:
                        return torch.zeros(layer.sampled_weight.shape[0])  # size=ch
                if isinstance(layer, nn.Linear) and layer.samples:
                    if layer.fisher is not None:
                        return torch.abs(layer.fisher.detach())
                    else:
                        return torch.zeros(layer.samples['weight'].shape[0])  # size=ch

        N = inputs.shape[0]
        split_data = 1
        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            network.zero_grad()
            outputs = network(inputs[st:en])
            if isinstance(outputs, tuple):
                outputs, loss_align = network(inputs[st:en])
            loss = loss_fn(outputs, targets[st:en])
            loss.backward()

        grads_abs_ch = get_layer_metric_array(network, mi, mode, pretrained=pretrained)

        # # broadcast channel value here to all parameters in that channel
        # # to be compatible with stuff downstream (which expects per-parameter metrics)
        # # TODO cleanup on the selectors/apply_prune_mask side (?)
        # # shapes = get_layer_metric_array(net, lambda l: l.weight.shape[1:], mode)
        # shapes = get_layer_metric_array(network, mi, mode, pretrained=pretrained)
        #
        # grads_abs = reshape_elements(grads_abs_ch, shapes, device)
        return grads_abs_ch

    mi_net = compute_mi(net)
    mi_pretrained = compute_mi(pretrained_model, pretrained=True)

    return mi_net, mi_pretrained, net.sample_embed_dim[0]