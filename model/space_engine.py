import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time
from lib.flops import count_flops
# from thop import profile
from .utils import output_metric
import numpy as np

from .utils import mode_hylite

lambda_align = 0.0 # no alignment loss

def sample_configs(choices):

    config = {}
    if mode_hylite:
        dimensions = ['mlp_ratio', 'num_heads', 'local_attn']
    else:
        dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, model_type, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, ewc=None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None):
    model.train()

    # criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if epoch == 0:
        if mode == 'retrain' and model_type == 'AUTOFORMER':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            # print(config)
            print(model_module.get_sampled_params_numel(config))
            print("FLOPS is {}".format(count_flops(model_module, input_shape=[200, 7, 7])))

    epoch_loss = 0.0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super' and model_type == 'AUTOFORMER':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain' and model_type == 'AUTOFORMER':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            # samples, targets = mixup_fn(samples, targets)
            samples, targets = samples,targets
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output, _ = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs, loss_align = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze()) + lambda_align * loss_align
                else:
                    outputs, loss_align = model(samples)
                    loss = criterion(outputs, targets) + lambda_align * loss_align
        else:
            outputs, loss_align = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output, _ = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze()) + lambda_align * loss_align
            else:
                loss = criterion(outputs, targets) + lambda_align * loss_align

        if ewc is not None:
            # ewc_loss = min(1e7 * ewc.penalty(model), 1.0)
            # ewc_loss = 1e6 * ewc.penalty(model)
            ewc_loss = ewc.penalty(model)
            print("loss: {:.4f}, ewc_loss: {}".format(loss, ewc_loss))
            loss += ewc_loss

        loss_value = loss.item()
        epoch_loss += loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    avg_epoch_loss = epoch_loss / len(data_loader) #average epoch loss over the number of batches
    print('Average epoch train loss: ', round(avg_epoch_loss, 4))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Epoch: {}, Averaged stats: {}".format(epoch, metric_logger))
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, round(avg_epoch_loss, 4)

@torch.no_grad()
def evaluate(data_loader, model_type, model, device, amp=True, choices=None, mode='super', retrain_config=None, for_hsi=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super' and model_type == 'AUTOFORMER':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    elif model_type == 'AUTOFORMER':
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config

    # if model_type == 'AUTOFORMER':
        # print("sampled model config: {}".format(config))
        # parameters = model_module.get_sampled_params_numel(config)
        # print("sampled model parameters: {}".format(parameters))

    tar = np.array([])
    pre = np.array([])
    epoch_loss = 0.0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output, loss_align = model(images)
                loss = criterion(output, target) + lambda_align * loss_align
        else:
            output, loss_align = model(images)
            loss = criterion(output, target) + lambda_align * loss_align

        epoch_loss += loss.item()

        ## for hyperspectral images
        pred = output.topk(1, 1, True, True)[1].t().squeeze()
        pre = np.append(pre, pred.data.cpu().numpy())
        tar = np.append(tar, target.data.cpu().numpy())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    avg_epoch_loss = epoch_loss / len(data_loader)
    if for_hsi:
        ## for hyperspectral images
        OA, AA, Kappa, AA2 = output_metric(tar, pre)
        print('Test OA: {:.4f}, AA: {:.4f}, Kappa: {:.4f}, loss: {:.4f}'.format(OA.item(), AA.item(), Kappa.item(),
                                                                   loss.item()))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, OA.item(), AA.item(), Kappa.item(), round(avg_epoch_loss, 4)
    else:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
