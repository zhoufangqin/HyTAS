import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from . import indicator
from ..p_utils import get_layer_metric_array, get_layer_metric_array_zico
import random

@indicator('attn_sim', bn=True, mode='param')
def compute_heads_sim_per_weight(net, inputs, targets, mode, loss_fn, split_data=1, pretrained_model=None):
    net.zero_grad()
    # inputs = torch.randn([16, 200, 49]).cuda()
    # inputs = torch.zeros([16, 200, 49]).cuda()
    N = inputs.shape[0]
    split_data = 1
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs, loss_align, all_attns = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    torch.cuda.synchronize()

    # across samples
    all_sim = []
    score = 0
    for layer in range(len(all_attns)):
        tmp = all_attns[layer]  # (samples, heads, 201,201)
        eigenvalues = torch.linalg.eigvalsh(tmp, UPLO='U') #(samples, heads, 201)
        # print(torch.sum(eigenvalues).item())
        # score += torch.sum(eigenvalues).item() #work not well, xx_attn_eigenvalues.csv

        eigenvalues = eigenvalues.reshape(eigenvalues.shape[0], -1).detach().cpu().numpy()
        nsr_std = np.std(eigenvalues, axis=0)  # across samples
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(eigenvalues), axis=0)
        tmp_sum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        print(np.log(tmp_sum))
        score += np.log(tmp_sum)

    # select the gradients that we want to use for search/prune
    def attn_sim(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, torch.nn.Conv2d) and layer._get_name() != 'PatchembedSuper':
            if layer.weight.grad is not None:
                return torch.abs(layer.weight.grad * layer.weight)
            else:
                return torch.zeros_like(layer.weight)
        if isinstance(layer, nn.Linear) and layer.samples:
            if layer.samples['weight'].grad is not None:
                # print(layer.samples['weight'].size())
                return torch.abs(layer.samples['weight'].grad * layer.samples['weight'])
            else:
                # print('None: ', layer.samples['weight'].size())
                return torch.zeros_like(layer.samples['weight'])

    grads_abs = get_layer_metric_array_zico(net, attn_sim, mode)


    print(score)
    if isinstance(score, np.ndarray):
        return score[0]
    else:
        return score
