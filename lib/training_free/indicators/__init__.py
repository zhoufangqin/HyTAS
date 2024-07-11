available_indicators = []
_indicator_impls = {}

def indicator(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def indicator_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_copy(bn=bn).to(device)
            else:
                net = net_orig
            if name =='NASWOT':
                ret = func(net, device, *args)
            elif name =='te_nas':
                ret = func(net)
            else:
                ret = func(net, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc
                import torch
                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _indicator_impls
        if name in _indicator_impls:
            raise KeyError(f'Duplicated indicator! {name}')
        available_indicators.append(name)
        _indicator_impls[name] = indicator_impl
        return func
    return make_impl


def calc_indicator(name, net, device, *args, **kwargs):
    return _indicator_impls[name](net, device, *args, **kwargs)

def load_all():
   from . import NASWOT
   from . import snip
   from . import grasp
   from . import te_nas
   from . import dss
   from . import NASWOT
   from . import zico
   from . import synflow
   from . import attn_sim
   from . import  logsynflow
   from . import grad_entropy
   from . import fisher
   from . import zico_act
   from . import mi
   from . import grad_norm
   from . import jacob_cov
   from . import snip_grad_norm
   from . import t_cet
   from . import croze
   from . import gradsign
   from . import cond_num

load_all()
