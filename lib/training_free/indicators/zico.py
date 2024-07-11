## code based on https://github.com/SLDGroup/ZiCo/blob/main/ZeroShotProxy/compute_zico.py
'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys
from . import indicator
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from ..p_utils import get_layer_metric_array_zico

@indicator('zico', bn=True, mode='param')
def getzico(network, trainloader, targets=None, mode=None, loss_fn=F.cross_entropy, split_data=1, pretrained_model=None):
    network.train()
    network.cuda()

    network.zero_grad()
    pretrained = False
    # N = trainloader.shape[0]
    # print('N: ', N) #64

    ''' Remark 3.1 Intuitively, Theorem. 3.1 tells us that the higher the gradient absolute mean across
    different training samples, the lower the training loss the model converges to; i.e., the network
    converges at a faster rate. Similarly, the smaller the gradient standard deviation across
    different training samples/batches, the lower the training loss the model can achieve
    '''
    if not pretrained:
        emb = network.sample_embed_dim[0]
    def zico(layer):
        if pretrained:
            if isinstance(layer, torch.nn.Linear):
                if layer.weight.grad is not None:
                    return (layer.weight.data.squeeze().detach().cpu().numpy(),
                            layer.weight.grad.data.detach().cpu().reshape(-1).numpy())
        else:
            if layer._get_name() == 'PatchembedSuper':
                if layer.sampled_weight.grad is not None:
                    return (layer.sampled_weight.data.squeeze().detach().cpu().numpy(), layer.sampled_weight.grad.data.detach().cpu().reshape(-1).numpy())

            if isinstance(layer, torch.nn.Linear):
                if layer.samples['weight'].grad is not None:
                    return (layer.samples['weight'].data.squeeze().detach().cpu().numpy(), layer.samples['weight'].grad.data.detach().cpu().reshape(-1).numpy())

    def compute_single_grad(samples, targets, pretrained=False):
        all_grads = []
        all_weights = []
        all_pred_var = []

        # print(samples.size())
        for i in range(samples.size(0)):

            if pretrained:
                output = network.forward(samples[i].unsqueeze(0))
                loss_align = 0
            else:
                output, loss_align, attn = network.forward(samples[i].unsqueeze(0))

            ## compute the predictive variance, try to minimize the predictive variance and maximize the predictive mean to be more general
            all_pred_var.append(output.softmax(dim=1))
            loss = loss_fn(output, targets[i].unsqueeze(0)) + loss_align #loss divided by network.sample_embed_dim[0] does not work
            loss.backward()
            grads = get_layer_metric_array_zico(network, zico, mode, pretrained=pretrained)
            grads_l = [grad[1] for grad in grads if grad is not None and grad[1] is not None]
            weights_l = [grad[0] for grad in grads if grad is not None and grad[0] is not None]
            # print(np.array(grads_new).shape) #23
            all_grads.append(np.array(grads_l))
            all_weights.append(np.array(weights_l))
            # nonlinearize(network, signs)

        all_pred_var = torch.stack(all_pred_var, dim=0)
        return all_grads, all_pred_var, all_weights

    # trainloader = torch.randn([16, 200, 49*3]).cuda()
    all_grads, all_pred_var, all_weights = compute_single_grad(trainloader, targets, pretrained=pretrained) #trainloader

    # nonlinearize(network, signs)

    if pretrained:
        return all_grads, all_pred_var, all_weights, 64
    else:
        return all_grads, all_pred_var, all_weights, network.sample_embed_dim