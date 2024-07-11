import numpy as np
import torch

from .p_utils import *
from . import indicators

import types
import copy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score

def normalize_to_01(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def no_op(self,x):
    return x

def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn==False:
        for l in net.modules():
            if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                l.forward = types.MethodType(no_op, l)
    return net

def find_indicators_arrays(net_orig, trainloader, dataload_info, device, indicator_names=None, loss_fn=F.cross_entropy, pretrained_model=None):
    if indicator_names is None:
        indicator_names = indicators.available_indicators

    dataload, num_imgs_or_batches, num_classes = dataload_info

    net_orig.to(device)
    if not hasattr(net_orig,'get_copy'):
        net_orig.get_copy = types.MethodType(copynet, net_orig)

    #move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu() 
    torch.cuda.empty_cache()

    #given 1 minibatch of data
    if dataload == 'random':
        inputs, targets = get_some_data(trainloader, num_batches=num_imgs_or_batches, device=device)
    elif dataload == 'grasp':
        inputs, targets = get_some_data_grasp(trainloader, num_classes, samples_per_class=num_imgs_or_batches, device=device)
    # elif dataload == 'zico':
    #     num_imgs_or_batches = 1
    #     # num_imgs_or_batches = len(trainloader)
    #     inputs, targets = get_some_data(trainloader, num_batches=num_imgs_or_batches, device=device)
    else:
        raise NotImplementedError(f'dataload {dataload} is not supported')

    done, ds = False, 10
    indicator_values = {}

    while not done:
        try:
            for indicator_name in indicator_names:
                if indicator_name not in indicator_values:
                    if indicator_name == 'NASWOT'  or indicator_name=='te_nas':
                        val = indicators.calc_indicator(indicator_name, net_orig, device, inputs)
                        indicator_values[indicator_name] = val
                    else:
                        val = indicators.calc_indicator(indicator_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds, pretrained_model=pretrained_model)
                        indicator_values[indicator_name] = val

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done=True
                if ds == inputs.shape[0]//2:
                    raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong') 
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
            else:
                raise e

    net_orig = net_orig.to(device).train()
    return indicator_values

def find_indicators(net_orig,
                  dataloader,
                  dataload_info,
                  device,
                  loss_fn=F.cross_entropy,
                  indicator_names=None,
                  indicators_arr=None, pretrained_model=None):
    

    def sum_arr(arrs):
        arr, emb = arrs
        if isinstance(emb, tuple):
            emb, heads = emb
        sum = 0.
        for i in range(len(arr)):
            if torch.all(arr[i] == 0):
                continue
            # print(arr[i].shape, torch.sum(arr[i]).item(), torch.sum(arr[i]).item()/emb)
            sum += torch.sum(arr[i])
            # sum += torch.log(torch.sum(arr[i]))
            # sum += torch.sum(arr[i]) / (arr[i].squeeze().shape[0] * arr[i].squeeze().shape[1]) #search_results_snip_norm.csv
        return sum.item()

    ## for single sample grad in a batch
    def zico_arr(all_grads):
        all_grads, all_preds_var, all_weights, emb_dim = all_grads
        all_grads = np.array(all_grads).T
        all_weights = np.array(all_weights).T
        # print(all_grads.shape) #layers,16 (embeding_layer, block(4 proj layers) * layer_num), conv*5, last layer
        layer_num = (all_grads.shape[0] - 2) // 4
        print("layer_num: ", layer_num)
        nsr_mean_sum_abs = 0
        for i in range(all_grads.shape[0]):  # across layers
            grads_s = []
            for j in range(all_grads.shape[1]):
                grads_s.append(all_grads[i][j].flatten()) #to compute the mean and var of gradients
                # grads_s.append(all_weights[i][j].flatten())  # to compute the mean and var of weights

            nsr_std = np.std(grads_s, axis=0)  # across samples
            nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(grads_s), axis=0)
            tmp_sum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            nsr_mean_sum_abs += np.log(tmp_sum)
        return round(np.log(nsr_mean_sum_abs), 4)

    def zico_acts(all_acts):
        all_act, emb_dim = all_acts
        nsr_mean_sum_abs = 0
        l = 0
        for i in range(len(all_act)):  # layers
            # print(all_act[i].shape, torch.sum(all_act[i]).item())
            if torch.all(all_act[i] == 0).item():
                continue
            nsr_std = np.std(all_act[i].cpu().numpy(), axis=0)  # across samples
            nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(all_act[i].cpu().numpy()), axis=0)
            tmp_sum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            # nsr_mean_sum_abs += np.log(tmp_sum)

            if i > 4 and i < len(all_act) - 1:
                nsr_mean_sum_abs += (1/(i-4)) * np.log(tmp_sum)
            else:
                nsr_mean_sum_abs += np.log(tmp_sum)
            l += 1

        return round(nsr_mean_sum_abs, 4)
        # return round(nsr_mean_sum_abs, 4) / (l+1) #indicated as xx_v4.cs

    def synflow_scores(v):
        arr, emb = v
        score = 0.

        ## the original code
        for grad_abs in arr:
            if torch.all(grad_abs == 0).item():
                continue
            if not torch.isnan(torch.sum(grad_abs)):
                print(grad_abs.shape, torch.sum(grad_abs).item())
                score += float(torch.sum(grad_abs)) #sum of all without taking average
            if len(grad_abs.shape) == 4:
                print(grad_abs.shape, torch.mean(torch.sum(grad_abs, dim=[1, 2, 3])).item())
                score += float(torch.mean(torch.sum(grad_abs, dim=[1, 2, 3])))
                # all_scores.append(float(torch.mean(torch.sum(grad_abs, dim=[1, 2, 3]))))
            elif len(grad_abs.shape) == 2:
                print(grad_abs.shape, torch.mean(torch.sum(grad_abs, dim=[1])).item())
                if not torch.isnan(torch.mean(torch.sum(grad_abs, dim=[1]))):
                    score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
                # all_scores.append(float(torch.mean(torch.sum(grad_abs, dim=[1]))))

        return score

    def cal_spectral_norm(arrs):
        ## if use gradients instead of weights, the mean and varience of the singular values are over 97% correlated to the embedding dimension

        ## Weight matrices with smaller spectral norms may lead to a more well-conditioned optimization problem,
        ## which can result in faster and more stable convergence during training.
        arr, emb = arrs
        sum = 0
        # ss = []
        l = 0
        for i, weights in enumerate(arr):

            if isinstance(weights, torch.Tensor):
                continue
            w, g = weights
            k = 32
            if torch.all(w==0):
                continue
            U, S, V = torch.svd(w)
            print(torch.sum(torch.abs(w)).item(), torch.max(S).item(), torch.min(S).item() / torch.max(S).item())
            # sum += torch.min(S) / torch.max(S) #the score has almost 0 relation to OA
            # sum += torch.max(S) #98% correlated to the embed_dim
            sum += torch.sum(torch.abs(w*g)) * (1- torch.max(S).item())
            l += 1

        return sum.item()

    def cal_mi(arrs):
        net_mi, pre_mi, emb = arrs
        sum = 0
        net_mi_new = []
        for i in range(len(net_mi)):
            if torch.all(net_mi[i] == 0).item():
                continue
            net_mi_new.append(net_mi[i])
        print(len(net_mi_new), len(pre_mi))  # 42, 22
        min_length = min(len(net_mi_new), len(pre_mi))-1
        net_mi_new = net_mi_new[:min_length] + net_mi_new[-1:]
        pre_mi = pre_mi[:min_length] + pre_mi[-1:]

        nets = []
        pres = []
        for s in range(net_mi_new[0].shape[0]):
            n = []
            p = []
            for i in range(len(pre_mi)):
                tmp_nets = net_mi_new[i].cpu().numpy()
                tmp_pre = pre_mi[i].cpu().numpy()
                n.extend(tmp_nets[s, :])
                p.extend(tmp_pre[s, :])
            nets.append(n)
            pres.append(p)
        print(np.array(nets).shape, np.array(pres).shape)
        return mutual_info_score(np.array(nets).flatten(), np.array(pres).flatten())

    def cal_t_cet(arrs):
        nwots_scores, snip_scores = arrs
        # print(len(nwots_scores), len(snip_scores)) #41, 42, because nwots_scores does not contain the patch_super_emb layer due to an error
        scores = []
        # for i in range(len(snip_scores)):
        for i in range(len(nwots_scores)+1):
            s = snip_scores[i].detach().view(-1)
            if torch.std(s) == 0:
                s = torch.sum(s)
            else:
                s = torch.sum(s) / torch.std(s)

            if s != 0:
                s = torch.log(s)

            # s = s * nwots_scores[i]
            if i >= 1:
                # print(s, nwots_scores[i-1])
                # s = nwots_scores[i - 1] / s #the result is much worse
                s = s * nwots_scores[i - 1]

            scores.append(s.cpu().numpy())
        return np.sum(scores)

    if indicators_arr is None:
        indicators_arr = find_indicators_arrays(net_orig, dataloader, dataload_info, device, loss_fn=loss_fn, indicator_names=indicator_names, pretrained_model=pretrained_model)

    indicators = {}
    for k,v in indicators_arr.items():
        if k == 'NASWOT' or k=='te_nas' or k=='gradsign':
            indicators[k] = v
        elif k == 'zico':
            indicators[k] = zico_arr(v)
        elif k == 'zico_act':
            indicators[k] = zico_acts(v)
        elif k == 'mi':
            indicators[k] = cal_mi(v)
        elif k == 'synflow' or k == 'logsynflow':
            indicators[k] = synflow_scores(v)
        elif k == 'attn_sim':
            indicators[k] = v
            # indicators[k] = sim_scores(v)
        elif k == 'grad_entropy':
            # indicators[k] = cal_grad_entrop(v) ## calculate the gradients entropy
            indicators[k] = cal_spectral_norm(v) ## calculate the spectral norm of weights
        elif k == 'jacob_cov':
            indicators[k] = v
        elif k == 't_cet':
            indicators[k] = cal_t_cet(v)
        else:
            indicators[k] = sum_arr(v)

    return indicators