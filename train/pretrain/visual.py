import datetime
import time
import warnings
from logging import Logger

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from timm.utils import AverageMeter

from data import build_loader
from models import build_model
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.utils import (auto_resume_helper, get_grad_norm, load_checkpoint,
                         reduce_tensor, save_checkpoint)

try:
    from apex import amp
except ImportError:
    amp = None

warnings.filterwarnings('ignore')


def pretrain_vis(cfg: DictConfig, logger: Logger):
    start_time = time.time()
    data_loader_train, data_loader_val, mixup_fn = build_loader(cfg)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"train size: {data_loader_train.source.total_size}\t"
                f"val size: {data_loader_val.source.total_size}"
                f"load time: {total_time_str}")

    logger.info(f"Creating model: {cfg.model.type}/{cfg.model.name}")
    model = build_model(cfg)
    model.cuda()
    logger.info(f"Model Arch -->\n{model}")

    optimizer = build_optimizer(cfg, model)
    if cfg.amp_opt_level != "O0":
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=cfg.amp_opt_level,
        )
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[cfg.dist.local_rank],
        broadcast_buffers=False,
    )
    model_without_ddp = model.module

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_params}")
    if hasattr(model_without_ddp, 'flops'):
        logger.info(f"number of GFLOPs: {model_without_ddp.flops() / 1e9}")

    lr_scheduler = build_scheduler(cfg, optimizer, len(data_loader_train))

    # TODO: implement criterion
    criterion = torch.nn.MSELoss()

    max_accuracy = 0.0

    if cfg.train.auto_resume:
        resume_file = auto_resume_helper(cfg.output, logger)
        if resume_file:
            if cfg.model.resume:
                logger.warning(
                    f"auto-resume changing resume file from {cfg.model.resume} to {resume_file}"
                )
            cfg.model.resume = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {cfg.output}, ignoring auto resume')

    if cfg.model.resume:
        max_accuracy = load_checkpoint(cfg, model_without_ddp, optimizer,
                                       lr_scheduler, logger)
        loss = validate(cfg, model, criterion, data_loader_val, logger)
        logger.info(
            f"Validation on {data_loader_val.source.total_size} samples:\t"
            f"Loss={loss:.4f}")
        if cfg.eval_mode:
            return

    if cfg.throughput_mode:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        data_loader_train.source.set_epoch(epoch)

        train_one_epoch(cfg, model, criterion, data_loader_train, optimizer,
                        epoch, lr_scheduler, logger)
        if dist.get_rank() == 0 and (epoch % cfg.save_freq == 0
                                     or epoch == (cfg.train.epochs - 1)):
            save_checkpoint(cfg, epoch, model_without_ddp, max_accuracy,
                            optimizer, lr_scheduler, logger)

        loss = validate(cfg, model, criterion, data_loader_val, logger)
        logger.info(
            f"Validation on {data_loader_val.source.total_size} samples:\t"
            f"Loss={loss:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')

    exit(0)

    epoch = 1
    t_start = time.time()
    for e in range(epoch):
        for i, (img, cap) in enumerate(data_loader_train):
            img = img.cpu()
            restore_image, mask_index = model(img)
            print(restore_image.size(), len(mask_index))
            print(f'iteration {i}', img.size(), cap.size())
    t_cost = time.time() - t_start
    print(
        f'time: {t_cost}, speed: {len(data_loader_train.source) * epoch / t_cost}imgs/s'
    )

    exit(0)


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch,
                    lr_scheduler, logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    data_time, batch_time, loss_meter, norm_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter()

    dist.barrier(device_ids=[config.local_rank])
    start = end = time.time()

    for idx, (imgs, _) in enumerate(data_loader):
        dist.barrier(device_ids=[config.local_rank])
        data_time.update(time.time() - end)

        restore_imgs, mask_index = model(imgs)
        loss = criterion(restore_imgs, imgs, mask_index)

        if config.train.accumulation_steps > 1:
            loss = loss / config.train.accumulation_steps

            if config.amp_opt_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                params = amp.master_params(optimizer)
            else:
                loss.backward()
                params = model.parameters()

            if config.train.clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, config.train.clip_grad)
            else:
                grad_norm = get_grad_norm(params)

            if (idx + 1) % config.train.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            if config.amp_opt_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                params = amp.master_params(optimizer)
            else:
                loss.backward()
                params = model.parameters()

            if config.train.clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, config.train.clip_grad)
            else:
                grad_norm = get_grad_norm(params)

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        dist.barrier(device_ids=[config.local_rank])

        loss_meter.update(loss.item(), imgs.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)

        end = time.time()

        if idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} (sec:{etas:.3f})  lr {lr:.7f}\t'
                f'batch_time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data_time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.1f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))} (sec:{epoch_time:.3f})"
    )


@torch.no_grad()
def validate(config, model, criterion, data_loader, logger):
    model.eval()
    batch_time, loss_meter = AverageMeter(), AverageMeter()

    end = time.time()
    for idx, (imgs, _) in enumerate(data_loader):
        restore_imgs, mask_index = model(imgs)
        loss = criterion(restore_imgs, imgs, mask_index)

        loss_meter.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Val Batch: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Mem {memory_used:.1f}MB')
    loss = reduce_tensor(loss_meter.avg)
    logger.info(f'* Loss {loss:.4f}')
    return loss


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for _, (imgs, _) in enumerate(data_loader):
        batch_size = imgs.shape[0]
        warmup_step, throughput_step = 50, 500
        timer = AverageMeter()
        for _ in range(warmup_step):
            model(imgs)
        logger.info(f"throughput averaged with {throughput_step} times")
        torch.cuda.synchronize()
        for _ in range(throughput_step):
            starter = torch.cuda.Event(enable_timing=True),
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            model(imgs)
            ender.record()
            torch.cuda.synchronize()
            timer.update(starter.elapsed_time(ender) / 1000)
        logger.info(
            f"batch_size {batch_size} throughput {batch_size / timer.avg}")
        return
