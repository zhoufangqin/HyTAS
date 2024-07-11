
import torch
import torch.nn as nn
import copy
import numpy as np
from . import indicator
import torch.nn.functional as F
import types
from ..p_utils import get_layer_metric_array

def safe_hooklogdet(K):
    s, ld = np.linalg.slogdet(K)
    return 0 if (np.isneginf(ld) and s== 0) else ld

@indicator('t_cet', bn=True, mode='param')
def compute_t_cet_per_layer(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):

    def get_t_cet_layerwise(net, inputs, targets):
        net = copy.deepcopy(net)
        net.eval()
        batch_size = inputs.shape[0]
        # model.K = torch.zeros(batch_size, batch_size).cuda()
        net.K_dict = {}

        def counting_forward_hook(module, inp, out):
            try:
                out = out.view(out.size(0), -1)
                x = (out > 0).float()
                K = x @ x.t()
                # print(x.cpu().numpy().sum(), module._get_name())
                if x.cpu().numpy().sum() == 0:
                    # model.K_dict[module.name] = 0
                    net.K_dict[module.alias] = 0
                else:
                    K2 = (1. - x) @ (1. - x.t())
                    matrix = K + K2
                    # model.K_dict[module.name] = hooklogdet(matrix.cpu().numpy())
                    abslogdet = safe_hooklogdet(matrix.cpu().numpy())
                    net.K_dict[module.alias] = 0. if np.isneginf(abslogdet) else abslogdet  # TODO: -inf
            except:
                pass

        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                module.alias = name
                module.register_forward_hook(counting_forward_hook)

        # inputs = inputs.cuda(device=device)
        # inputs = inputs.to(device=device)
        with torch.no_grad():
            net(inputs)

        scores = []
        for name, module in net.named_modules():
            # if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            if isinstance(module, nn.Linear):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= net.sample_layer_num:
                    continue
                scores.append(net.K_dict[name])

        # scores = copy.deepcopy(scores)
        del net
        del inputs
        return scores

    def snip_forward_conv2d(self, x):
        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        return F.conv2d(x, self.sampled_weight * self.weight_mask, self.sampled_bias,
                        stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(
            2).transpose(1, 2)
    def snip_forward_linear(self, x):
        return F.linear(x, self.samples['weight'] * self.weight_mask, self.samples['bias'])

    def compute_snip_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
        for layer in net.modules():
            if layer._get_name() == 'PatchembedSuper':
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.sampled_weight))
                layer.sampled_weight = layer.sampled_weight.detach()
            if isinstance(layer, nn.Linear) and layer.samples:  # is 10 the number of classes?
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.samples['weight']))
                layer.samples['weight'] = layer.samples['weight'].detach()

            if layer._get_name() == 'PatchembedSuper':
                layer.forward = types.MethodType(snip_forward_conv2d, layer)
            if isinstance(layer, nn.Linear) and layer.samples:
                layer.forward = types.MethodType(snip_forward_linear, layer)

        net.zero_grad()
        N = inputs.shape[0]
        split_data = 1
        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            outputs, loss_align = net.forward(inputs[st:en])
            loss = loss_fn(outputs, targets[st:en])
            loss.backward()

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

        return grads_abs

    nwots_scores = get_t_cet_layerwise(net, inputs, targets)
    snip_scores = compute_snip_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None)
    return (nwots_scores, snip_scores)