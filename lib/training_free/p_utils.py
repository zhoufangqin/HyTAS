import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict

def get_some_data(train_dataloader, num_batches, device): #get input and label
    traindata = []
    dataloader_iter = iter(train_dataloader)
    # print(len(train_dataloader)) #1 when distributed=False
    for _ in range(min(num_batches, len(train_dataloader))): ## use a minibatch
    # for _ in range(len(train_dataloader)): ## use the full training set
        traindata.append(next(dataloader_iter))
    inputs  = torch.cat([a for a,_ in traindata])
    targets = torch.cat([b for _,b in traindata])
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    # print(inputs.size(), targets.size()) #torch.Size([16, 200, 49]) torch.Size([16])
    return inputs, targets

def get_some_data_grasp(train_dataloader, num_classes, samples_per_class, device):
    # print(len(train_dataloader), num_classes, samples_per_class) #1,16,1
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(train_dataloader)
    # while True:
    inputs, targets = next(dataloader_iter)
    for idx in range(inputs.shape[0]):
        x, y = inputs[idx:idx+1], targets[idx:idx+1]
        category = y.item()
        if len(datas[category]) == samples_per_class:
            mark[category] = True
            continue
        datas[category].append(x)
        labels[category].append(y)
        # if len(mark) == num_classes:
        #     break

    x = torch.cat([torch.cat(_, 0) for _ in datas]).to(device) 
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1).to(device)
    return x, y

# def get_layer_metric_array_adv_feats(net, advnet, feats, adv_feats, metric, mode):
def get_layer_metric_array_adv_feats(net, advnet, metric, mode):
    metric_array = []
    # layer_cnt = 0

    for layer, layer_adv in zip(net.modules(), advnet.modules()):
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #     # layer_cnt += 1
        #     # metric_array.append(metric(layer, layer_adv, feats[layer_cnt], adv_feats[layer_cnt]))
        #     metric_array.append(metric(layer, layer_adv))
        if layer._get_name() == 'PatchembedSuper':
            metric_array.append(metric(layer, layer_adv))
        if isinstance(layer, nn.Linear) and layer.samples:
            metric_array.append(metric(layer, layer_adv))

    return metric_array

def get_flattened_metric(net, metric, mode):
    grad_list = []
    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if layer._get_name() == 'PatchembedSuper':
            grad_list.append(metric(layer).flatten())
        if isinstance(layer, nn.Linear) and layer.samples:
            grad_list.append(metric(layer).flatten())
    flattened_grad = np.concatenate(grad_list)

    return flattened_grad

def get_layer_metric_array(net, metric, mode, pretrained=False):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if pretrained:
            if isinstance(layer, nn.Linear):
                metric_array.append(metric(layer))
        else:
            if layer._get_name() == 'PatchembedSuper':
                metric_array.append(metric(layer))
            if isinstance(layer, nn.Linear) and layer.samples:
                metric_array.append(metric(layer))

    return metric_array

def get_layer_metric_array_dss(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue

        if layer._get_name() == 'PatchembedSuper':
            metric_array.append(metric(layer))
        ######
        if isinstance(layer, nn.Linear) and layer.samples:
            metric_array.append(metric(layer))
        if isinstance(layer, torch.nn.Linear) and layer.out_features == 16: #need to change
            metric_array.append(metric(layer))
    return metric_array

def get_layer_metric_array_dss_hsi(net, metric, mode):
    metric_array = []

    for name, layer in net.named_modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue

        if name == 'patch_embed_super':
            metric_array.append(metric(layer))
        if len(name.split('.'))==4 and (name.split('.')[-1] == 'qkv' or name.split('.')[-1]=='proj') and isinstance(layer, nn.Linear) and layer.samples:
            metric_array.append(metric(layer, name='msa'))
        if len(name.split('.'))==3 and (name.split('.')[-1] =='fc1' or name.split('.')[-1]=='fc2') and isinstance(layer, nn.Linear) and layer.samples:
            metric_array.append(metric(layer, name='mlp'))
        if name == 'head' and isinstance(layer, torch.nn.Linear):
            metric_array.append(metric(layer, name='head'))
    return metric_array

def get_layer_metric_array_zico(net, metric, mode, pretrained=False):
    metric_array = []
    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if pretrained:
            if isinstance(layer, nn.Linear):
                metric_array.append(metric(layer))
        else:
            if layer._get_name() == 'PatchembedSuper':
                metric_array.append(metric(layer))
            if isinstance(layer, nn.Linear) and layer.samples:
                metric_array.append(metric(layer))
            # if isinstance(layer, nn.Conv2d) and layer._get_name() != 'PatchembedSuper':
            #     metric_array.append(metric(layer))

    return metric_array

def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e,sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads
    if type(elements[0]) == list:
        outer = []
        for e,sh in zip(elements, shapes):
            outer.append(broadcast_val(e,sh))
        return outer
    else:
        return broadcast_val(elements, shapes)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### from https://github.com/HyeonjeongHa/CRoZe/blob/main/zero_cost_methods/pruners/p_utils.py#L32
EPS = 1E-30
def adj_weights(
        model,
        inputs,
        targets,
        w=0.1,
        case=0,
        loss_maximize=False,
):
    proxy_net = copy.deepcopy(model)
    proxy_net.train()
    proxy_optim = torch.optim.SGD(proxy_net.parameters(), lr=0.001)
    # proxy_net.double()

    if targets is None:
        # loss = torch.sum(proxy_net(inputs.double()))
        loss = torch.sum(proxy_net(inputs))
    else:
        outs = proxy_net(inputs)
        if isinstance(outs, tuple):
            outs, _ = outs
        loss = F.cross_entropy(outs, targets)

    if loss_maximize:
        loss = -1 * loss

    proxy_optim.zero_grad()
    try:
        loss.backward()
    except:
        return None
    proxy_optim.step()

    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy_net.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w

    names_in_diff = diff_dict.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(w * diff_dict[name])

    del proxy_net, proxy_optim
    return model

