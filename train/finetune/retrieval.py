from logging import Logger

from omegaconf import DictConfig


def finetune_retrieval(cfg: DictConfig, logger: Logger):
    logger.info(f'{cfg}')
