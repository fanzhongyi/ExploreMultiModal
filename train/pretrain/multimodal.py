'''
MultiModal Pretraining.

    NOTE:
        1. MLM, ITC, hard sample ITM.

    FIXME:
        1.

    TODO:
        1. HyperParameter Tuning.
        2. Momentum Update & Negative Queue

'''

import datetime
import json
import math
import os
import time
import warnings
from logging import Logger
from typing import Iterable

import torch
from data.multitask_datamodule import MTDataModule
from models import build_model
from omegaconf import DictConfig, OmegaConf
from utils import utils
from utils.lr_scheduler import build_scheduler
from utils.optim_factory import create_optimizer
from utils.utils import NativeScalerWithGradNormCount as NativeScaler

warnings.filterwarnings('ignore')


def pretrain_mum(cfg: DictConfig, logger: Logger):

    if cfg.deepspeed.enable:
        try:
            import deepspeed
            ds_init = deepspeed.initialize
        except Exception:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    logger.info(f"Creating model: {cfg.model.type}/{cfg.model.name}")
    model = build_model(cfg)
    logger.debug(f"Model Arch -->\n{model}")

    model_without_ddp = model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_params / 1e6} M")

    patch_size = model.transformer.patch_size
    logger.info(f"Patch size = {patch_size}")

    model.cuda()
    if cfg.deepspeed.enable:
        if cfg.deepspeed.pth2ds is not None:
            torch_ckpt = torch.load(cfg.deepspeed.pth2ds, map_location='cpu')
            model.load_from_ckpt(torch_ckpt)

        model, optimizer, _, _ = ds_init(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            dist_init_required=not cfg.dist.distributed,
            config=OmegaConf.to_object(cfg.deepspeed.config),
        )
        model_without_ddp = model.module
        loss_scaler = None

        if cfg.deepspeed.pth2ds is not None:
            cfg.train.auto_resume = False
            logger.warning(
                f'Init Deepspeed with torch state_dict: {cfg.deepspeed.pth2ds},'
                f' and disable auto resume ckpt as default.')

    else:
        if cfg.dist.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[cfg.dist.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            model_without_ddp = model.module
        optimizer = create_optimizer(
            cfg.train,
            model_without_ddp,
            logger=logger,
        )
        loss_scaler = NativeScaler()

    logger.info(
        f'Model Memory:\t{torch.cuda.max_memory_allocated() / 2 ** 20} MB')

    dm = MTDataModule(cfg)
    data_loader_train = dm.train_dataloader()
    data_loader_val = dm.val_dataloader()
    # data_loader_test = dm.test_dataloader()

    logger.info(f"Train size = {len(data_loader_train.dataset)}")
    logger.info(f"Val size = {len(data_loader_val.dataset)}")
    # logger.info(f"Test size = {len(data_loader_test.dataset)}")

    num_steps_per_epoch = len(data_loader_train)
    total_batch_size = cfg.data.batch_size * cfg.dist.world_size
    # cfg.train.base_lr = cfg.train.base_lr * total_batch_size / 256

    logger.info(f"LR = {cfg.train.base_lr:.8f}")
    logger.info(f"Total batch size = {total_batch_size}")
    logger.info(f"Train steps per epoch = {num_steps_per_epoch}")

    lr_scheduler = build_scheduler(cfg, optimizer, num_steps_per_epoch)

    utils.auto_load_model(
        cfg=cfg,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_scaler=loss_scaler,
        logger=logger,
    )

    if cfg.throughput_mode:
        throughput(model, data_loader_val, logger)
        return

    if cfg.eval_mode:
        assert len(cfg.train.resume) > 0, 'eval_mode need ckpt!'

    if cfg.eval_mode or cfg.train.start_epoch == cfg.train.epochs:
        evaluate(model, data_loader_val, logger=logger, cfg=cfg)
        return

    if cfg.dist.rank == 0 and cfg.wandb.enable:
        wb_logger = utils.WandbLogger(cfg)
        wb_logger.wandb.watch(model_without_ddp, log='all', log_graph=True)
        wb_logger.update_config(OmegaConf.to_object(cfg))
    else:
        wb_logger = None

    logger.info("Start training")
    start_time = time.time()

    best_epoch, best_loss = -1, 100

    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):

        data_loader_train.sampler.set_epoch(epoch)
        cfg.train.cur_epoch = epoch

        if wb_logger is not None:
            wb_logger.set_step(epoch * num_steps_per_epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            epoch,
            loss_scaler,
            max_norm=cfg.train.clip_grad,
            start_steps=epoch * num_steps_per_epoch,
            lr_scheduler=lr_scheduler,
            logger=logger,
            wb_logger=wb_logger,
            cfg=cfg,
        )

        if epoch % cfg.train.save_freq == 0 or epoch + 1 == cfg.train.epochs:
            utils.save_model(
                cfg=cfg,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
            eval_stats = evaluate(model,
                                  data_loader_val,
                                  logger=logger,
                                  cfg=cfg)

            if wb_logger is not None:
                wb_logger.log(
                    head='val/metric',
                    epoch=epoch,
                    step=(epoch + 1) * num_steps_per_epoch,
                    **eval_stats,
                )

            if eval_stats['loss'] < best_loss:
                best_epoch, best_loss = epoch, eval_stats['loss']
                logger.info(f'ckpt-{best_epoch} achieve new loss {best_loss}')
            else:
                logger.info(f'ckpt-{best_epoch} keeps best loss {best_loss}')

            utils.remove_models(cfg, epoch, best_epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_params': n_params,
        }

        if utils.is_main_process():
            with open(os.path.join(cfg.output_dir, "log_stats.json"),
                      mode="a",
                      encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    logger.info(
        f'the ckpt-{best_epoch} achieve best loss {best_loss} on val set.')
    cfg.minimize_metric = best_loss

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')

    if wb_logger is not None and cfg.wandb.alert:
        wb_logger.alert(
            'MultiModal Pretrain end',
            f'the ckpt-{best_epoch} achieve best loss {best_loss} on val set.')
    if wb_logger is not None:
        wb_logger.finish()


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    lr_scheduler=None,
                    start_steps=None,
                    logger=None,
                    wb_logger=None,
                    cfg=None):

    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'[Epoch {epoch}]'
    print_freq = cfg.train.print_freq
    print_stat_level = cfg.train.print_stat_level

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    grad_acc_steps = cfg.train.accumulation_steps

    for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, logger, header)):

        it = start_steps + step  # global training iteration

        if loss_scaler is None:
            batch_fp16 = dict()
            for k, v in batch.items():
                if v.type() in ['torch.FloatTensor', 'torch.cuda.FloatTensor']:
                    batch_fp16[k] = v.half()
                else:
                    batch_fp16[k] = v
            outputs = model(batch_fp16)
        else:
            with torch.cuda.amp.autocast():
                outputs = model(batch)

        total_loss = sum([
            v for k, v in outputs.items()
            if "task_loss" in k and math.isfinite(v)
        ])
        # __import__('ipdb').set_trace()

        for k, v in outputs.items():
            if 'loss' in k and not math.isfinite(v):
                logger.warning(f"{k} is {v}, but not stop training")
                logger.warning(f"\n{outputs}")
                torch.save(
                    outputs,
                    os.path.join(cfg.output_dir,
                                 f"{cfg.dist.rank}_{it}_nan_obj.pth"))

        if not math.isfinite(total_loss):
            logger.warning(f"Loss is {total_loss}, stopping training")
            import sys
            sys.exit(1)

        total_loss4backward = total_loss
        if cfg.train.flat_loss:
            total_loss4backward = sum([
                v / v.detach()
                for k, v in outputs.items()
                if "task_loss" in k and math.isfinite(v)
            ])

        total_loss4backward = total_loss4backward / grad_acc_steps

        if loss_scaler is None:
            model.backward(total_loss4backward)
            model.step()
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(
                total_loss4backward,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(step + 1) % grad_acc_steps == 0,
            )
            if (step + 1) % grad_acc_steps == 0:
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        if lr_scheduler is not None:
            lr_scheduler.step_update(it)

        # torch.cuda.synchronize()
        metrics = dict()

        metrics.update(dict(loss=total_loss, loss_scale=loss_scale_value))
        metrics.update(grad_norm=grad_norm)

        if print_stat_level >= 1:
            task_loss = {k: v for k, v in outputs.items() if 'task_loss' in k}
            metrics.update(task_loss)
        if print_stat_level >= 2:
            fine_loss = {k: v for k, v in outputs.items() if 'Loss' in k}
            metrics.update(fine_loss)

        if 'mlm_mean_acc' in outputs.keys():
            metrics.update(mlm_acc=dict(value=outputs['mlm_mean_acc'].item(),
                                        n=outputs['mlm_count']),)
        if 'mim_mean_acc' in outputs.keys():
            metrics.update(mim_acc=dict(value=outputs['mim_mean_acc'].item(),
                                        n=outputs['mim_count']),)
        if 'itm_mean_acc' in outputs.keys():
            metrics.update(itm_acc=dict(value=outputs['itm_mean_acc'].item(),
                                        n=outputs['itm_count']),)
        if 'itc_temp' in outputs.keys():
            metrics.update(itc_temp=outputs['itc_temp'])
            metrics.update(itc_i2t_acc=dict(
                value=outputs['itc_i2t_mean_acc'].item(),
                n=outputs['itc_i2t_count']),)
            metrics.update(itc_t2i_acc=dict(
                value=outputs['itc_t2i_mean_acc'].item(),
                n=outputs['itc_t2i_count']),)

        opts = dict()

        min_lr, max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        opts.update(dict(lr=max_lr, min_lr=min_lr))

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        opts.update(weight_decay=weight_decay_value)

        metric_logger.update(**metrics)
        metric_logger.update(**opts)

        if wb_logger is not None and it % print_freq == 0 and it / print_freq > 3. and (
                it + 1) % grad_acc_steps == 0:
            for k in metrics.keys():
                if isinstance(metrics[k], dict):
                    metrics[k] = metrics[k]['value']
            wb_logger.log(head='train/metric',
                          step=it // grad_acc_steps,
                          **metrics)
            wb_logger.log(head='train/opt', step=it // grad_acc_steps, **opts)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             data_loader: Iterable,
             logger=None,
             cfg=None):

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[Evaluate]'

    print_freq = cfg.train.print_freq * 2
    print_stat_level = cfg.train.print_stat_level

    for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, logger, header)):

        with torch.cuda.amp.autocast():
            outputs = model(batch)

        total_loss = sum([v for k, v in outputs.items() if "task_loss" in k])
        # __import__('ipdb').set_trace()

        metrics = dict()

        metrics.update(loss=total_loss)

        if print_stat_level >= 1:
            task_loss = {k: v for k, v in outputs.items() if 'task_loss' in k}
            metrics.update(task_loss)
        if print_stat_level >= 2:
            fine_loss = {k: v for k, v in outputs.items() if 'Loss' in k}
            metrics.update(fine_loss)

        if 'mlm_mean_acc' in outputs.keys():
            metrics.update(mlm_acc=dict(value=outputs['mlm_mean_acc'].item(),
                                        n=outputs['mlm_count']),)
        if 'itm_mean_acc' in outputs.keys():
            metrics.update(itm_acc=dict(value=outputs['itm_mean_acc'].item(),
                                        n=outputs['itm_count']),)
        if 'mim_mean_acc' in outputs.keys():
            metrics.update(mim_acc=dict(value=outputs['mim_mean_acc'].item(),
                                        n=outputs['mim_count']),)
        if 'itc_temp' in outputs.keys():
            metrics.update(itc_i2t_acc=dict(
                value=outputs['itc_i2t_mean_acc'].item(),
                n=outputs['itc_i2t_count']),)
            metrics.update(itc_t2i_acc=dict(
                value=outputs['itc_t2i_mean_acc'].item(),
                n=outputs['itc_t2i_count']),)

        metric_logger.update(**metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def throughput(model, data_loader, logger):

    model.eval()
    for _, batch in enumerate(data_loader):

        print(batch.keys())

        batch_size = batch['image'].size(0)
        warmup_step, throughput_step = 100, 1000
        timer = utils.SmoothedValue(window_size=100)
        for _ in range(warmup_step):
            model(batch)
        logger.info(f"testing throughput with {throughput_step} times")
        torch.cuda.synchronize()
        for _ in range(throughput_step):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            model(batch)
            ender.record()
            torch.cuda.synchronize()
            timer.update(starter.elapsed_time(ender) / 1000)
        logger.info(f"{batch_size=} throughput {batch_size/timer.global_avg}")
        return
