import json

import torch
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from torch import optim as optim

try:
    from apex.optimizers import FusedAdam, FusedLAMB, FusedNovoGrad, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_parameter_groups(
        model,
        base_lr,
        lr_mult_head,
        lr_mult_fusion,
        weight_decay=1e-5,
        skip_list=(),
):

    fusion_layer = model.config.model.fusion_layer
    depth = model.config.model.depth

    fusion_names = [f"blocks.{i}" for i in range(fusion_layer, depth)]
    fusion_names.append('pooler')
    head_names = [
        "mlm_head",
        "itc_head",
        "itm_head",
        "mim_head",
        "vqa_classifier",
        "nlvr2_classifier",
        "snli_classifier",
    ]

    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) <= 1 or name.endswith(".bias") or name in skip_list:
            decay_name = "no_decay"
            this_weight_decay = 0.
        else:
            decay_name = "decay"
            this_weight_decay = weight_decay

        if any(hd in name for hd in head_names):
            part_name = "head_layer"
            lr = base_lr * lr_mult_head
        elif any(fl in name for fl in fusion_names):
            part_name = "fusion_layer"
            lr = base_lr * lr_mult_fusion
        else:
            part_name = "bottom_layer"
            lr = base_lr

        group_name = f"{part_name}_{decay_name}"

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "params": [],
                "weight_decay": this_weight_decay,
                "lr": lr
            }
            parameter_group_vars[group_name] = {
                "params": [],
                "weight_decay": this_weight_decay,
                "lr": lr
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(cfg, model, skip_list=None):

    opt_lower = cfg.opt.name.lower()
    weight_decay = cfg.weight_decay

    skip = skip_list or {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    parameters = get_parameter_groups(model,
                                      base_lr=cfg.base_lr,
                                      lr_mult_head=cfg.lr_mult_head,
                                      lr_mult_fusion=cfg.lr_mult_fusion,
                                      weight_decay=weight_decay,
                                      skip_list=skip)
    weight_decay = 0.

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(
        ), 'APEX & CUDA required for fused optimizers'

    opt_args = dict(lr=cfg.base_lr, weight_decay=weight_decay)
    opt_args['eps'] = cfg.opt.eps
    opt_args['betas'] = cfg.opt.betas

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters,
                              momentum=cfg.opt.momentum,
                              nesterov=True,
                              **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters,
                              momentum=cfg.opt.momentum,
                              nesterov=False,
                              **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters,
                         momentum=cfg.opt.momentum,
                         nesterov=True,
                         **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not cfg.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters,
                                  alpha=0.9,
                                  momentum=cfg.opt.momentum,
                                  **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters,
                              alpha=0.9,
                              momentum=cfg.opt.momentum,
                              **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters,
                             momentum=cfg.opt.momentum,
                             nesterov=True,
                             **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters,
                             momentum=cfg.opt.momentum,
                             nesterov=False,
                             **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
