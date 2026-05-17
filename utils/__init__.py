# coding: utf-8

"""
General utility functions.
"""
import os
import random
import numpy as np
import torch
import datetime


def get_local_time():
    """Get current time as string."""
    cur = datetime.datetime.now()
    return cur.strftime('%b-%d-%Y-%H-%M-%S')


def init_seed(seed):
    """Initialize random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    """
    Validation-based early stopping.

    Args:
        value: current result
        best: best result
        cur_step: number of consecutive steps that did not exceed the best result
        max_step: threshold steps for stopping
        bigger: whether the bigger the better

    Returns:
        tuple: (best, cur_step, stop_flag, update_flag)
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    """Convert result dict to string."""
    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str