# coding: utf-8

"""
Node classification task head placeholder.
For future extension to support node classification tasks on multimodal graphs.
"""
from .base_task import BaseTask


class NodeClassificationTask(BaseTask):
    """
    Task head for node classification.
    Placeholder for future implementation.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        raise NotImplementedError('NodeClassificationTask is not yet implemented.')

    def train_epoch(self, train_data, epoch_idx, optimizer, loss_func):
        raise NotImplementedError('NodeClassificationTask is not yet implemented.')

    def evaluate(self, eval_data, is_test=False):
        raise NotImplementedError('NodeClassificationTask is not yet implemented.')

    def predict(self, interaction):
        raise NotImplementedError('NodeClassificationTask is not yet implemented.')