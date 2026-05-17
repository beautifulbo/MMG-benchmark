# coding: utf-8

"""
MMG-benchmark tasks module.
"""
from .base_task import BaseTask
from .link_prediction import LinkPredictionTask
from .node_classification import NodeClassificationTask
from .metrics import metrics_dict

__all__ = [
    'BaseTask',
    'LinkPredictionTask',
    'NodeClassificationTask',
    'metrics_dict',
]