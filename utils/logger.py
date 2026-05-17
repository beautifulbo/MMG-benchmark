# coding: utf-8

"""
Logger utility for MMG-benchmark.
"""
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO, WARNING, ERROR
from logging import StreamHandler, FileHandler


def init_logger(config):
    """Initialize logger with config."""
    log_dir = config.get('log_dir', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, '{}-{}.log'.format(
        config['model'], config['dataset']
    ))

    level = INFO

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)

    basicConfig(
        level=level,
        format=format_str,
        handlers=[
            FileHandler(log_file, encoding='utf-8'),
            StreamHandler(sys.stdout)
        ]
    )


import logging