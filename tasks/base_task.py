# coding: utf-8

"""
Base task head for multimodal graph learning tasks.
"""
import torch.nn as nn


class BaseTask(nn.Module):
    """
    Base class for all task heads.
    All tasks should implement train_epoch(), eval_epoch(), and predict() methods.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = config['device']

    def train_epoch(self, train_data, epoch_idx):
        """
        Train for one epoch.

        Args:
            train_data: Training data loader
            epoch_idx: Current epoch index

        Returns:
            tuple: (total_loss, loss_batches)
        """
        raise NotImplementedError('Method [train_epoch] should be implemented.')

    def evaluate(self, eval_data, is_test=False):
        """
        Evaluate the model.

        Args:
            eval_data: Evaluation data loader
            is_test: Whether this is test evaluation

        Returns:
            dict: Evaluation metrics
        """
        raise NotImplementedError('Method [evaluate] should be implemented.')

    def predict(self, interaction):
        """
        Make predictions.

        Args:
            interaction: Input interaction

        Returns:
            torch.Tensor: Predicted scores
        """
        raise NotImplementedError('Method [predict] should be implemented.')