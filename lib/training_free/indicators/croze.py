import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import copy
from . import indicator
from ..p_utils import adj_weights, get_layer_metric_array, get_layer_metric_array_adv_feats


def fisher_forward_conv2d(self, x):
    x = x.unsqueeze(2).permute(0, 3, 2, 1)
    x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size,
                 padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1, 2)
    # intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act
def fisher_forward_linear_samples(self, x):
    x = F.linear(x, self.samples['weight'], self.samples['bias'])
    self.act = self.dummy(x)
    return self.act

def fgsm_attack(net, image, target, epsilon):
    perturbed_image = image.detach().clone()
    perturbed_image.requires_grad = True
    net.zero_grad()

    logits = net(perturbed_image)
    if isinstance(logits, tuple):
        logits, _ = logits
    loss = F.cross_entropy(logits, target)
    loss.backward()

    sign_data_grad = perturbed_image.grad.sign_()
    perturbed_image = perturbed_image - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# class ActivationExtractor:
#     def __init__(self, model):
#         self.model = model
#         self.activations = {}
#         self.layers = []
#
#         # Register hooks for specified layers
#         for name, layer in model.named_modules():
#             # if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
#             if isinstance(layer, nn.Linear):
#                 if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= model.sample_layer_num:
#                     continue
#                 # if layer.samples:
#                 #     layer.forward = types.MethodType(fisher_forward_linear_samples, layer)
#                 layer.register_forward_hook(self.hook_fn(name))
#                 self.layers.append(layer)
#
#     def hook_fn(self, layer_name):
#         def hook(module, input, output):
#             self.activations[layer_name] = output.detach()
#         return hook
#
#     def extract_activations(self, inputs):
#         self.activations = {layer_name: None for layer_name in self.layers}
#         self.model(inputs)
#         return self.activations

@indicator('croze', bn=False, mode='param')
def compute_croze_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
    origin_inputs, origin_outputs = inputs, targets

    cos_loss = nn.CosineSimilarity(dim=0)
    ce_loss = nn.CrossEntropyLoss()

    def reg_hook(net):
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or layer._get_name() == 'PatchembedSuper':
                layer.act = 0.
                layer.acts = None
                layer.dummy = nn.Identity()
                if layer._get_name() == 'PatchembedSuper':
                    layer.forward = types.MethodType(fisher_forward_conv2d, layer)
                if isinstance(layer, nn.Linear):
                    if layer.samples:
                        layer.forward = types.MethodType(fisher_forward_linear_samples, layer)

                def hook_factory(layer):
                    def hook(module, grad_input, grad_output):
                        layer.acts = layer.act
                        del layer.act
                    return hook
                # register backward hook on identity fcn to compute fisher info
                layer.hook_handle = layer.dummy.register_backward_hook(hook_factory(layer))
    def unreg_hook(net):
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or layer._get_name() == 'PatchembedSuper':
                # del layer.acts
                layer.hook_handle.remove()

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
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    advnet = copy.deepcopy(net)

    # keep signs of all params
    signs = linearize(net)
    adv_signs = linearize(advnet)

    net.train()
    # activation_extractor_origin = ActivationExtractor(net)
    reg_hook(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    # net.double()
    # advnet.double()

    # output = net.forward(origin_inputs.double())
    output = net.forward(origin_inputs)
    if isinstance(output, tuple):
        output, _ = output
    output.retain_grad()

    # activations_origin = activation_extractor_origin.extract_activations(origin_inputs)

    # advnet = adj_weights(advnet, origin_inputs.double(), origin_outputs, 2.0, loss_maximize=True)
    # advinput = fgsm_attack(advnet, origin_inputs.double(), origin_outputs, 0.01)
    advnet = adj_weights(advnet, origin_inputs, origin_outputs, 2.0, loss_maximize=True)
    advinput = fgsm_attack(advnet, origin_inputs, origin_outputs, 0.01)

    advnet.train()
    reg_hook(advnet)
    # activation_extractor_ad = ActivationExtractor(advnet)
    # advnet.zero_grad()
    adv_outputs = advnet.forward(advinput.detach())
    if isinstance(adv_outputs, tuple):
        adv_outputs, _ = adv_outputs
    adv_outputs.retain_grad()

    # activations_ad = activation_extractor_ad.extract_activations(advinput.detach())

    loss = ce_loss(output, origin_outputs) + ce_loss(adv_outputs, origin_outputs)
    loss.backward()

    def croze(layer, layer_adv):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None and layer_adv.sampled_weight.grad is not None:
                w_sim = (1 + cos_loss(layer_adv.sampled_weight, layer.sampled_weight)).sum()
                sim = (torch.abs(cos_loss(layer_adv.sampled_weight.grad, layer.sampled_weight.grad))).sum()
                feat_sim = (1 + cos_loss(layer_adv.acts, layer.acts)).sum()
                # return torch.abs(w_sim * sim * feat_sim)
                ret = torch.log(torch.abs(w_sim * sim * feat_sim))
            else:
                ret = torch.zeros_like(layer.sampled_weight)
            del layer.acts
            del layer_adv.acts
            layer.hook_handle.remove()
            layer_adv.hook_handle.remove()
            return ret
        if isinstance(layer, torch.nn.Linear) and layer.samples:
            if layer.samples['weight'].grad is not None and layer_adv.samples['weight'].grad is not None:
                w_sim = (1 + cos_loss(layer_adv.samples['weight'], layer.samples['weight'])).sum()
                sim = (torch.abs(cos_loss(layer_adv.samples['weight'].grad, layer.samples['weight'].grad))).sum()
                feat_sim = (1 + cos_loss(layer_adv.acts, layer.acts)).sum()
                # return torch.abs(w_sim * sim * feat_sim)
                # print(w_sim, sim, feat_sim)
                ret = torch.log(torch.abs(w_sim * sim * feat_sim))
            else:
                ret = torch.zeros_like(layer.samples['weight'])
            del layer.acts
            del layer_adv.acts
            layer.hook_handle.remove()
            layer_adv.hook_handle.remove()
            return ret
    grads_abs = get_layer_metric_array_adv_feats(net, advnet, croze, mode)
    # grads_abs = get_layer_metric_array(net, croze, mode)

    # apply signs of all params
    nonlinearize(net, signs)
    nonlinearize(advnet, adv_signs)

    unreg_hook(net)
    unreg_hook(advnet)
    del advnet

    return grads_abs, net.sample_embed_dim[0]