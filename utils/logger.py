import functools
import logging
import os
import sys

from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir,
                  dist_rank=0,
                  name='',
                  log_level='info',
                  extra_tag=''):
    # create logger
    logger = logging.getLogger(name)
    if log_level == 'info':
        logger.setLevel(logging.INFO)
    elif log_level == 'debug':
        logger.setLevel(logging.DEBUG)

    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s -> %(funcName)s %(lineno)d): %(levelname)s %(message)s'

    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s -> %(funcName)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f'log_rank{dist_rank}_{extra_tag}.txt'),
        mode='a',
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
