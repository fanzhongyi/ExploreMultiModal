"""
The Entry of ExploreMultiModal Project

Project TODO :
==================================
    0. under reconstruction.
    1. Implement vision encoder & decoder with MAE train strategy. Done!!
    2. Implement txt encoder & decoder with MLM & LM. Doing~~
    3. Implement fastest img loading with DALI. Done!!
    4. Implement fastest txt loading. Next...
    5. Implement multi-modal pretrain task with MAE & MLP & ITM. Next...
    6. Implement finetune task one-by-one. NNNNext...
    7. debug hyperparameters forever...
==================================
"""

import os
import random
import subprocess
import warnings
from logging import Logger

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from train.finetune.caption import finetune_caption
from train.finetune.inpainting import finetune_inpainting
from train.finetune.nlvr2 import finetune_nlvr2
from train.finetune.ref import finetune_ref
from train.finetune.retrieval import finetune_retrieval
from train.finetune.vqa import finetune_vqa
from train.pretrain.multimodal import pretrain_mum
from train.pretrain.text import pretrain_txt
from train.pretrain.visual import pretrain_vis
from utils.logger import create_logger

try:
    from apex import amp
except ImportError:
    amp = None

warnings.filterwarnings('ignore')


def main(cfg: DictConfig, logger: Logger):

    train_phase = cfg.train.phase

    # TODO:
    #  1. Implement vision encoder & decoder with MAE train strategy. Done!!
    #  2. Implement txt encoder & decoder with MLM & LM. Doing~~
    #  3. Implement fastest img loading with DALI. Done!!
    #  4. Implement fastest txt loading. Next...
    #  5. Implement multi-modal pretrain task with MAE & MLP & ITM. Next...
    #  6. Implement finetune task one-by-one. NNNNext...
    #  7. debug hyperparameters forever...

    # Pretrain Group

    if train_phase == 'pretrain_vis':
        pretrain_vis(cfg, logger)
    elif train_phase == 'pretrain_txt':
        pretrain_txt(cfg, logger)
    elif train_phase == 'pretrain_mum':
        pretrain_mum(cfg, logger)

    # discriminative group

    elif train_phase == 'finetune_retrieval':
        finetune_retrieval(cfg, logger)
    elif train_phase == 'finetune_ref':
        finetune_ref(cfg, logger)
    elif train_phase == 'finetune_nlvr2':
        finetune_nlvr2(cfg, logger)
    # semi-generative task
    elif train_phase == 'finetune_vqa':
        finetune_vqa(cfg, logger)

    # generative group

    # img -> txt
    elif train_phase == 'finetune_caption':
        finetune_caption(cfg, logger)
    # txt -> img
    elif train_phase == 'finetune_inpainting':
        finetune_inpainting(cfg, logger)


@hydra.main(config_path='conf', config_name='config')
def setup(cfg: DictConfig) -> None:
    if cfg.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    if "SLURM_JOB_ID" in os.environ:
        cfg.dist.SLURM_ENABLE = True
        rank = int(os.getenv("SLURM_PROCID", 0))
        world_size = int(os.getenv("SLURM_NTASKS", 1))
        local_rank = int(os.getenv("SLURM_LOCALID", 0))
        local_world_size = int(os.getenv("SLURM_NTASKS_PER_NODE", 1))
        slurm_nodelist = os.getenv("SLURM_NODELIST", '127.0.0.1')
        master_addr = subprocess.getoutput(
            f"bash -c 'scontrol show hostname {cfg.DIST.SLURM_NODELIST} | head -n1'"
        )
        cfg.dist.slurm_nodelist = slurm_nodelist
        cfg.dist.master_addr = master_addr
    else:
        rank = int(os.getenv('RANK', 0))
        world_size = int(os.getenv('WORLD_SIZE', 1))
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 1))

    cfg.dist.rank = rank
    cfg.dist.world_size = world_size
    cfg.dist.local_rank = local_rank
    cfg.dist.local_world_size = local_world_size

    print(
        f"ENV: {rank = }, {world_size = }, {local_rank = }, {local_world_size = }"
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    dist.barrier(device_ids=[local_rank])

    seed = cfg.seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    linear_scaled_lr = cfg.train.base_lr * cfg.data.batch_size * dist.get_world_size(
    ) / 512.0
    linear_scaled_warmup_lr = cfg.train.warmup_lr * cfg.data.batch_size * dist.get_world_size(
    ) / 512.0
    linear_scaled_min_lr = cfg.train.min_lr * cfg.data.batch_size * dist.get_world_size(
    ) / 512.0
    if cfg.train.accumulation_steps > 1:
        linear_scaled_lr *= cfg.train.accumulation_steps
        linear_scaled_warmup_lr *= cfg.train.accumulation_steps
        linear_scaled_min_lr *= cfg.train.accumulation_steps

    cfg.train.base_lr = linear_scaled_lr
    cfg.train.warmup_lr = linear_scaled_warmup_lr
    cfg.train.min_lr = linear_scaled_min_lr

    os.makedirs(cfg.output, exist_ok=True)
    logger = create_logger(
        output_dir=cfg.output,
        dist_rank=dist.get_rank(),
        name=f"{cfg.model.name}",
    )
    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, "cfg.yaml")
        OmegaConf.save(cfg, path)
        logger.info(f"Full cfg saved to {path}")

    logger.info(f"CONFIG INFO -->\n{OmegaConf.to_yaml(cfg)}")
    logger.info(
        f"ENV: {rank = }, {world_size = }, {local_rank = }, {local_world_size = }"
    )
    logger.debug(f'Process ENV -->\n{os.environ}')

    if cfg.ipdb:
        from ipdb import launch_ipdb_on_exception
        with launch_ipdb_on_exception():
            logger.warning("Launch ipdb on exception!")
            main(cfg, logger)
    else:
        main(cfg, logger)


if __name__ == "__main__":
    setup()
