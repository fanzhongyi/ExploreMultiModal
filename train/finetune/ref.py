from logging import Logger

from omegaconf import DictConfig


def finetune_ref(cfg: DictConfig, logger: Logger):
    logger.info(f'{cfg}')
