from logging import Logger

from omegaconf import DictConfig


def finetune_nlvr2(cfg: DictConfig, logger: Logger):
    logger.info(f'{cfg}')


def finetune_caption(cfg: DictConfig, logger: Logger):
    logger.info(f'{cfg}')


def finetune_inpainting(cfg: DictConfig, logger: Logger):
    logger.info(f'{cfg}')
