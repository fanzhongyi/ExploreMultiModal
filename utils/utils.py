import datetime
import io
import json
import math
import os
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from timm.utils import get_state_dict
from torch._six import inf


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
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64,
                         device='cuda')
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
        return self.fmt.format(median=self.median,
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
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, dict):
                self.meters[k].update(**v)
            else:
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
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, logger, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
            'time: {time}', 'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(i,
                                       len(iterable),
                                       eta=eta_string,
                                       meters=str(self),
                                       time=str(iter_time),
                                       data=str(data_time),
                                       memory=torch.cuda.max_memory_reserved() /
                                       MB))
                else:
                    logger.info(
                        log_msg.format(i,
                                       len(iterable),
                                       eta=eta_string,
                                       meters=str(self),
                                       time=str(iter_time),
                                       data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            f'{header} Total time: {total_time_str} ({total_time/len(iterable):.4f} s/it)'
        )


class WandbLogger(object):

    def __init__(self, config):
        os.environ["WANDB_API_KEY"] = config.wandb.token
        os.environ["WANDB_HOST"] = config.wandb.host
        os.environ["WANDB_MODE"] = config.wandb.mode

        self.wandb = wandb
        os.makedirs(f"{config.output_dir}/wandb", exist_ok=True)

        self.run = wandb.init(
            config=config,
            dir=f"{config.output_dir}",
            id="-".join([config.train.phase, config.tag]),
            name=config.wandb.name,
            project=config.wandb.project,
        )
        # self.wandb.run.log_code(
        #     '.',
        #     include_fn=lambda path: path.endswith(".py") or path.endswith(
        #         ".yaml"),
        # )

        self.step = 0
        self.meters = []

    def update_config(self, config):
        self.wandb.config.update(config, allow_val_change=True)

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def log(
        self,
        head='train',
        step=None,
        commit=False,
        **kwargs,
    ):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                key = f'{head}/{k}'
                self.meters.append(key)
                if 'loss' in key:
                    self.wandb.define_metric(
                        key,
                        summary='min',
                        goal='minimize',
                    )
                if 'acc' in key or 'score' in key:
                    self.wandb.define_metric(
                        key,
                        summary='max',
                        goal='maximize',
                    )

        self.set_step(step)
        self.wandb.log(
            {f'{head}/{k}': v for k, v in kwargs.items()},
            step=self.step,
            commit=commit,
        )

    def alert(self, title, text):
        self.wandb.alert(
            title=title,
            text=text,
            level=wandb.AlertLevel.INFO,
            wait_duration=10,
        )

    def finish(self):
        self.wandb.finish()


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
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def dist_barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(cfg_dist):
    if 'SLURM_PROCID' in os.environ:
        cfg_dist.slurm_enable = True
        cfg_dist.slurm_job_id = int(os.environ['SLURM_JOBID'])
        cfg_dist.slurm_nodelist = os.environ["SLURM_NODELIST"]
        cfg_dist.master_addr = os.environ['MASTER_ADDR']
        cfg_dist.rank = int(os.environ['SLURM_PROCID'])
        cfg_dist.world_size = int(os.environ["SLURM_NTASKS"])
        cfg_dist.local_rank = int(os.environ["SLURM_LOCALID"])
        cfg_dist.local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg_dist.rank = int(os.environ["RANK"])
        cfg_dist.world_size = int(os.environ['WORLD_SIZE'])
        cfg_dist.local_rank = int(os.environ['LOCAL_RANK'])
        # cfg_dist.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    else:
        print('Not using distributed mode')
        cfg_dist.distributed = False
        cfg_dist.rank = 0
        cfg_dist.world_size = 1
        cfg_dist.local_rank = 0
        cfg_dist.local_world_size = 1
        return

    cfg_dist.distributed = True
    torch.cuda.set_device(cfg_dist.local_rank)
    cfg_dist.dist_backend = 'nccl'
    print(
        f'| dist init (rank {cfg_dist.rank}): {cfg_dist.dist_url}, local_rank {cfg_dist.local_rank}',
        flush=True)
    dist.init_process_group(backend=cfg_dist.dist_backend,
                            init_method=cfg_dist.dist_url,
                            world_size=cfg_dist.world_size,
                            rank=cfg_dist.rank)
    dist.barrier()
    #  setup_for_distributed(cfg_dist.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self,
                 loss,
                 optimizer,
                 clip_grad=None,
                 parameters=None,
                 create_graph=False,
                 update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(
            p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters
            ]), norm_type)
    return total_norm


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(
        optimizer, "loss_scale") else optimizer.cur_scale


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([
        final_value + 0.5 * (base_value - final_value) *
        (1 + math.cos(math.pi * i / (len(iters)))) for i in iters
    ])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def load_state_dict(model,
                    state_dict,
                    prefix='',
                    ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".
              format(model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(cfg,
               epoch,
               model,
               model_without_ddp,
               optimizer,
               lr_scheduler,
               loss_scaler,
               model_ema=None):
    output_dir = Path(cfg.output_dir)
    suffix = '.pth' if loss_scaler is not None else '.ds'
    ckpt_name = f'checkpoint-{epoch}' + suffix

    if loss_scaler is not None:
        checkpoint_path = output_dir / ckpt_name
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'cfg': cfg,
        }
        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)
        save_on_master(to_save, checkpoint_path)
    else:
        # deepspeed
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=cfg.output_dir,
                              tag=ckpt_name,
                              client_state=client_state)
        # non deepspeed adaption, only used with non zero optimization
        save_on_master({'model': model_without_ddp.state_dict()},
                       output_dir / ckpt_name.replace('.ds', '.model'))
    return ckpt_name


def remove_models(cfg, epoch, best_epoch):
    if cfg.dist.rank == 0:
        output_dir = Path(cfg.output_dir)

        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit() and t not in [str(epoch), str(best_epoch)]:
                shutil.rmtree(ckpt) if os.path.isdir(ckpt) else os.remove(ckpt)


# XXX: Cyclomatic complexity too high
def auto_load_model(cfg,
                    model,
                    model_without_ddp,
                    optimizer,
                    lr_scheduler,
                    loss_scaler,
                    model_ema=None,
                    logger=None):

    exp_dir = Path(cfg.exp_dir)

    if loss_scaler is not None:
        ckpt_pattern = 'checkpoint-%d.pth'
    else:
        ckpt_pattern = 'checkpoint-%d.ds'

    if cfg.train.auto_resume and len(cfg.train.resume) == 0:
        import glob
        all_checkpoints = glob.glob(
            os.path.join(exp_dir, '*', ckpt_pattern.replace("%d", "*")))
        latest_ckpt = -1
        latest_ckpt_path = ''
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit() and int(t) > latest_ckpt:
                latest_ckpt = int(t)
                latest_ckpt_path = ckpt
        if latest_ckpt >= 0 and len(latest_ckpt_path) > 0:
            cfg.train.resume = latest_ckpt_path
        logger.warning(f"Auto resume checkpoint: {cfg.train.resume}")

    if loss_scaler is not None:

        # torch.amp
        if cfg.train.resume:
            if cfg.train.resume.startswith('https'):
                ckpt = torch.hub.load_state_dict_from_url(cfg.train.resume,
                                                          map_location='cpu',
                                                          check_hash=True)
            else:
                ckpt = torch.load(cfg.train.resume, map_location='cpu')

            match, is_beit = model_without_ddp.load_from_ckpt(ckpt['model'])

            if is_beit:
                logger.warning(
                    f"Initialized BEiT pretrained => {cfg.train.resume}")
            else:
                logger.info(f"Resume checkpoint ==> {cfg.train.resume}")

            if len(match.missing_keys) > 0:
                logger.warning(
                    f"Weights not initialized from pretrained model: {match.missing_keys}"
                )
            if len(match.unexpected_keys) > 0:
                logger.warning(
                    f"Weights from pretrained model not used: {match.unexpected_keys}"
                )

            if 'cfg' in ckpt:
                ckpt_cfg = ckpt['cfg']
                if (cfg.train.phase, cfg.tag) == (ckpt_cfg.train.phase,
                                                  ckpt_cfg.tag):
                    if 'optimizer' in ckpt and 'lr_scheduler' in ckpt and 'epoch' in ckpt:
                        cfg.train.start_epoch = ckpt['epoch'] + 1
                        optimizer.load_state_dict(ckpt['optimizer'])
                        ckpt_total_epochs = ckpt['cfg'].train.epochs

                        if cfg.train.start_epoch < ckpt_total_epochs:
                            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

                        if cfg.model_ema:
                            _load_checkpoint_for_ema(model_ema,
                                                     ckpt['model_ema'])
                        if 'scaler' in ckpt:
                            loss_scaler.load_state_dict(ckpt['scaler'])
                        logger.info("Load states with optim & sched!")
        else:
            logger.info('No ckpt or BEiT, start training from stratch...')

    else:

        # deepspeed
        if cfg.train.resume:
            if os.path.isdir(cfg.train.resume):
                _, client_states = model.load_checkpoint(
                    os.path.dirname(cfg.train.resume),
                    tag=os.path.basename(cfg.train.resume))
                cfg.train.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if cfg.model_ema:
                        _load_checkpoint_for_ema(model_ema,
                                                 client_states['model_ema'])
            else:
                ckpt = torch.load(cfg.train.resume, map_location='cpu')
                match, is_beit = model_without_ddp.load_from_ckpt(ckpt['model'])

                if is_beit:
                    logger.warning(
                        f"Initialized BEiT pretrained => {cfg.train.resume}")
                else:
                    logger.info(f"Resume checkpoint ==> {cfg.train.resume}")

                if len(match.missing_keys) > 0:
                    logger.warning(
                        f"Weights not initialized from pretrained model: {match.missing_keys}"
                    )
                if len(match.unexpected_keys) > 0:
                    logger.warning(
                        f"Weights from pretrained model not used: {match.unexpected_keys}"
                    )
        else:
            logger.info(
                'No ckpt or BEiT, training from stratch with deepspeed...')
