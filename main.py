"""
The Entry of ExploreMultiModal Project

Project TODO :
==================================
    0. under reconstruction.
    1. Implement vision encoder & decoder with MAE train strategy. Reimplement.
    2. Implement txt encoder & decoder with MLM & LM. Doing~~
    3. Implement fastest img loading with DALI. Back to orignal loader...
    4. Implement fastest txt loading. No Idea...
    5. Implement multi-modal pretrain task with MAE & MLP & ITM. Next...
    6. Implement finetune task one-by-one. NNNNext...
    7. debug hyperparameters forever...
==================================
"""

import os
import random
import time
import warnings
from logging import Logger

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

from train.finetune.caption import finetune_caption
from train.finetune.inpainting import finetune_inpainting
from train.finetune.nlvr2 import finetune_nlvr2
from train.finetune.ref import finetune_ref
from train.finetune.retrieval import finetune_retrieval
from train.finetune.visual import finetune_vis
from train.finetune.vqa import finetune_vqa
from train.pretrain.multimodal import pretrain_mum
from train.pretrain.text import pretrain_txt
from train.pretrain.visual import pretrain_vis
from utils import utils
from utils.logger import create_logger

warnings.filterwarnings('ignore')


def main(cfg: DictConfig, logger: Logger):

    phase = cfg.train.phase
    logger.info(f"============ train_phase: {cfg.train.phase} ==============")

    # Pretrain Group

    if phase == 'pretrain_vis':
        pretrain_vis(cfg, logger)
    elif phase == 'pretrain_txt':
        pretrain_txt(cfg, logger)
    elif phase == 'pretrain_mum':
        pretrain_mum(cfg, logger)

    # Mono modal finetune for validating

    elif phase == 'finetune_vis':
        finetune_vis(cfg, logger)

    # discriminative group

    elif phase == 'finetune_retrieval':
        finetune_retrieval(cfg, logger)
    elif phase == 'finetune_ref':
        finetune_ref(cfg, logger)
    elif phase == 'finetune_nlvr2':
        finetune_nlvr2(cfg, logger)
    # semi-generative task
    elif phase == 'finetune_vqa':
        finetune_vqa(cfg, logger)

    # generative group

    # img -> txt
    elif phase == 'finetune_caption':
        finetune_caption(cfg, logger)
    # txt -> img
    elif phase == 'finetune_inpainting':
        finetune_inpainting(cfg, logger)


@hydra.main(config_path='conf', config_name='config')
def setup(cfg: DictConfig) -> None:

    utils.init_distributed_mode(cfg.dist)

    seed = cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    exp_time = time.strftime('%Y-%m-%dH%H:%M:%S', time.localtime())

    cfg.exp_dir = os.path.join(cfg.output_dir, cfg.train.phase, cfg.model.name,
                               cfg.tag)
    cfg.output_dir = os.path.join(cfg.exp_dir, exp_time)
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = create_logger(
        output_dir=cfg.output_dir,
        dist_rank=utils.get_rank(),
        name=f"{cfg.model.name} | {cfg.train.phase}",
        log_level=cfg.log_level,
        extra_tag=cfg.dist.slurm_nodename,
    )

    code_dir = os.path.join(cfg.output_dir, f'code_id_{cfg.dist.slurm_job_id}')
    os.makedirs(code_dir, exist_ok=True)
    if utils.get_rank() == 0:
        os.system(f' find . -path "**/output/**" -prune '
                  f' -name "*.yaml" -o -name "*.py" '
                  f' -type f | xargs tar -czvf {code_dir}/code.tar.gz ')

    if utils.get_rank() == 0:
        path = os.path.join(cfg.output_dir, "cfg.yaml")
        OmegaConf.save(cfg, path)
        logger.info(f"Full cfg saved to {path}, may be modified later...")

    logger.info(f"CONFIG INFO -->\n{OmegaConf.to_yaml(cfg)}")
    logger.debug(f'Process ENV -->\n{os.environ}')

    if cfg.ipdb:
        from ipdb import launch_ipdb_on_exception
        with launch_ipdb_on_exception():
            logger.warning("Launch ipdb on exception!")
            main(cfg, logger)
    else:
        main(cfg, logger)

    if utils.get_rank() == 0:
        path = os.path.join(cfg.output_dir, "cfg_final.yaml")
        OmegaConf.save(cfg, path)
        logger.info(f"Final cfg saved to {path}, check with previous version.")


if __name__ == "__main__":
    setup()
