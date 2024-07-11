import torch

from . import indicator
from ..p_utils import get_layer_metric_array_dss, get_layer_metric_array_dss_hsi
import torch.nn as nn

@indicator('dss', bn=False, mode='param')
def compute_dss_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None, pretrained_model=None):
# compute per weight,only use "net","inputs","mode"
    device = inputs.device #decide the device

    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param) # change the tensor in to [1,-1,...] weight&bias
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)

    net.zero_grad() #initialize the grad to '0'
    input_dim = list(inputs[0,:].shape) #dimension of input
    inputs = torch.ones([1] + input_dim).float().to(device) #a tensor contains only '1'
    output, _ = net.forward(inputs) #output = the output after go through the forward
    torch.sum(output).backward()

    # emb = net.sample_embed_dim[0]
    def dss(layer, name=None):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and name=='msa' and layer.samples:
            if layer.samples['weight'].grad is not None:
                mul = torch.abs(
                    torch.norm(layer.samples['weight'].grad, 'nuc') * torch.norm(layer.samples['weight'], 'nuc'))
                return mul
            else:
                return torch.zeros_like(layer.samples['weight'])
        if isinstance(layer, nn.Linear) and name == 'mlp' and layer.samples:
            if layer.samples['weight'].grad is not None:
                mul = torch.abs(layer.samples['weight'].grad * layer.samples['weight'])
                return mul
            else:
                return torch.zeros_like(layer.samples['weight'])
        if isinstance(layer, torch.nn.Linear) and name == 'head':
            if layer.samples['weight'].grad is not None:
                mul = torch.abs(layer.samples['weight'].grad * layer.samples['weight'])
                return mul

            else:
                return torch.zeros_like(layer.samples['weight'])

    # grads_abs = get_layer_metric_array_dss(net, dss, mode)
    grads_abs = get_layer_metric_array_dss_hsi(net, dss, mode)

    nonlinearize(net, signs)

    return grads_abs, net.sample_embed_dim[0]


