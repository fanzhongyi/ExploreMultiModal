from logging import Logger

from omegaconf import DictConfig


def pretrain_mum(cfg: DictConfig, logger: Logger):
    logger.info(f'{cfg}')
