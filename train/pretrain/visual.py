"""
Visual Pretraining

NOTE: load checkpoints before create dataset, for dali adaptation.

"""

import datetime
import json
import math
import os
import sys
import time
import warnings
from logging import Logger
from typing import Iterable

import torch
import torch.nn as nn
import wandb
from einops import rearrange
from models import build_model
from omegaconf import DictConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import utils
from utils.optim_factory import create_optimizer
from utils.utils import NativeScalerWithGradNormCount as NativeScaler

warnings.filterwarnings('ignore')


def pretrain_vis(cfg: DictConfig, logger: Logger):

    logger.info(f"Creating model: {cfg.model.type}/{cfg.model.name}")
    model = build_model(cfg)
    logger.debug(f"Model Arch -->\n{model}")

    model_without_ddp = model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_params / 1e6} M")

    patch_size = model.encoder.patch_embed.patch_size
    cfg.model.patch_size = patch_size
    cfg.model.window_size = (cfg.model.img_size // patch_size[0],
                             cfg.model.img_size // patch_size[1])
    logger.info(f"Patch size = {patch_size}")

    model.cuda()
    if cfg.dist.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.dist.local_rank],
            broadcast_buffers=False,
            # find_unused_parameters=True,
        )
        model_without_ddp = model.module
        logger.info(
            f'model memory_used:\t{torch.cuda.max_memory_allocated() / 2 ** 20} MB'
        )

    optimizer = create_optimizer(cfg.train, model_without_ddp)
    loss_scaler = NativeScaler()

    # NOTE: load model and start_epoch before creating dali data_loader.
    utils.auto_load_model(
        cfg=cfg,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    build_pretrain_visual_loader = ...

    data_loader_train, data_loader_val, mixup_fn = build_pretrain_visual_loader(
        cfg)
    logger.info(f"train size: {data_loader_train.source.total_size}\t"
                f"val size: {data_loader_val.source.total_size}")

    num_training_steps_per_epoch = len(data_loader_train)
    total_batch_size = cfg.data.batch_size * cfg.dist.world_size
    cfg.train.base_lr = cfg.train.base_lr * total_batch_size / 256

    logger.info(f"LR = {cfg.train.base_lr:.8f}")
    logger.info(f"Total batch size = {total_batch_size}")
    logger.info(f"Number of training steps = {num_training_steps_per_epoch}")
    logger.info(
        f"Number of training examples per epoch = {total_batch_size * num_training_steps_per_epoch}"
    )

    lr_schedule_values = utils.cosine_scheduler(
        cfg.train.base_lr,
        cfg.train.min_lr,
        cfg.train.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=cfg.train.warmup_epochs,
        warmup_steps=cfg.train.warmup_steps,
    )
    wd_schedule_values = utils.cosine_scheduler(
        cfg.train.weight_decay,
        cfg.train.weight_decay_end,
        cfg.train.epochs,
        num_training_steps_per_epoch,
    )
    logger.info(f"Max WD = {max(wd_schedule_values):.7f}, "
                f"Min WD = {min(wd_schedule_values):.7f}")

    if cfg.throughput_mode:
        throughput(model, data_loader_val, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            epoch,
            loss_scaler,
            max_norm=cfg.train.clip_grad,
            patch_size=patch_size[0],
            normlize_target=cfg.train.normlize_target,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            logger=logger,
        )

        if epoch % cfg.train.save_freq == 0 or epoch + 1 == cfg.train.epochs:
            utils.save_model(
                cfg=cfg,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

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

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    patch_size: int = 16,
                    normlize_target: bool = True,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    logger=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'[Epoch {epoch}]'
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, logger, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group[
                        "lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos = batch
        bool_masked_pos = bool_masked_pos.to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN,
                                   device=torch.device('cuda'))[None, :, None,
                                                                None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD,
                                  device=torch.device('cuda'))[None, :, None,
                                                               None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(
                    unnorm_images,
                    'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                    p1=patch_size,
                    p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(
                    dim=-2, keepdim=True)) / (images_squeeze.var(
                        dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                    unnorm_images,
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_size,
                    p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(images, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.warning(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss,
                                optimizer,
                                clip_grad=max_norm,
                                parameters=model.parameters(),
                                create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def throughput(model, data_loader, logger):

    model.eval()
    for _, batch in enumerate(data_loader):
        images, bool_masked_pos = batch
        bool_masked_pos = bool_masked_pos.to(torch.bool)
        batch_size = images.shape[0]
        warmup_step, throughput_step = 50, 100
        timer = utils.SmoothedValue(window_size=100)
        for _ in range(warmup_step):
            model(images, bool_masked_pos)
        logger.info(f"throughput averaged with {throughput_step} times")
        torch.cuda.synchronize()
        for _ in range(throughput_step):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            model(images, bool_masked_pos)
            ender.record()
            torch.cuda.synchronize()
            timer.update(starter.elapsed_time(ender) / 1000)
        logger.info(f"{batch_size=} throughput {batch_size/timer.global_avg}")
