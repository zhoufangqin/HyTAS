import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import pandas as pd
from timm.utils.model import unwrap_model
from .flops import count_flops

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # if torch.cuda.is_available():
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time),
                #         memory=torch.cuda.max_memory_allocated() / MB))
                # else:
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('{} Total time: {} ({:.4f} s / it)'.format(
        #     header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        args.gpu = args.rank % torch.cuda.device_count()
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        # print(args.rank, args.world_size, args.gpu)
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def save_retrain_results(file_path, retrain_cfg, max_OA, OA, AA, Kappa, retrain_time, num_params_subnet, flops,
                         train_losses=None, test_losses=None, test_oas=None, mean_variances_weights=None, mean_variances_grads=None, grads_layer_norm=None):
    retrain_cfg_df = pd.DataFrame([retrain_cfg])
    if train_losses and test_losses and test_oas and mean_variances_weights and mean_variances_grads and grads_layer_norm:
        res = pd.DataFrame([{"max_OA": max_OA, "OA": OA, "AA": AA, "Kappa": Kappa, 'retrain_time': retrain_time, 'num_params_subnet': num_params_subnet,
                             'flops': flops, 'train_losses': train_losses, 'test_losses': test_losses, 'test_oas': test_oas,
                             'mean_variances_weights': mean_variances_weights, 'mean_variances_grads': mean_variances_grads, 'grads_layer_norm': grads_layer_norm}])
    else:
        res = pd.DataFrame([{"max_OA": max_OA, "OA": OA, "AA": AA, "Kappa": Kappa, 'retrain_time': retrain_time,
                             'num_params_subnet': num_params_subnet,
                             'flops': flops}])
    df_all = pd.concat([retrain_cfg_df, res], axis=1)
    if os.path.isfile(file_path):
        df_all.to_csv(file_path, mode='a', index=True, header=False, float_format='%.4f')
    else:
        df_all.to_csv(file_path, index=True, float_format='%.4f')

def check_retrained(file_path, retrain_cfg):
    retrain_cfg = pd.DataFrame([retrain_cfg])
    retrain_cfg[['mlp_ratio', 'num_heads', 'embed_dim']] = retrain_cfg[['mlp_ratio', 'num_heads', 'embed_dim']].astype(str)
    if os.path.isfile(file_path):
        retrain_results = pd.read_csv(file_path)[['mlp_ratio', 'num_heads', 'embed_dim', 'layer_num']]
        tmp = pd.merge(retrain_results, retrain_cfg, on=['mlp_ratio', 'num_heads', 'embed_dim', 'layer_num'], how='inner')
        if len(tmp) == 0:
            print('The sampled architecture is not trained yet')
            return False
        else:
            return True
    else:
        print('The retrain_results file is not exist.')
        return False


def count_para_flops(model, retrain_config, input_shape=[200,7,7]):
    config = retrain_config
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=config)
    # print(config)
    num_params_subnet = model_module.get_sampled_params_numel(config)
    flops = count_flops(model_module, input_shape=input_shape)
    # print()
    # print("FLOPS is {}".format(count_flops(model_module, input_shape=[200, 7, 7])))
    return num_params_subnet, flops

import yaml
def replace_local_attn():
    folder_path = './experiments/HyLITE/fix0/indian100'
    for filename in os.listdir(folder_path):
        if filename.endswith(".yaml"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
                yaml_data['RETRAIN']['LOCAL_ATTN'] = [x if x == 0 else 0 for x in yaml_data['RETRAIN']['LOCAL_ATTN']]

                ## for deleting local_attn
                # del yaml_data['RETRAIN']['LOCAL_ATTN']
                # del yaml_data['SEARCH_SPACE']['LOCAL_ATTN']
                # del yaml_data['SUPERNET']['LOCAL_ATTN']

                ## for adding local-attn
                # yaml_data['RETRAIN']['LOCAL_ATTN'] = [0 for i in range(yaml_data['RETRAIN']['DEPTH'])]
                # yaml_data['SEARCH_SPACE']['LOCAL_ATTN'] = [0, 1]
                # yaml_data['SUPERNET']['LOCAL_ATTN'] = 1
            with open(file_path, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file)
