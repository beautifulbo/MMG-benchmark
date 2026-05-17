# coding: utf-8

"""
Abstract recommender base classes.
"""
import os
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    """
    Base class for all models.
    """

    def pre_epoch_processing(self):
        """Called before each training epoch."""
        pass

    def post_epoch_processing(self):
        """Called after each training epoch."""
        pass

    def calculate_loss(self, interaction):
        """
        Calculate the training loss for a batch data.

        Args:
            interaction: Interaction batch

        Returns:
            torch.Tensor: Training loss
        """
        raise NotImplementedError

    def predict(self, interaction):
        """
        Predict the scores between users and items.

        Args:
            interaction: Interaction batch

        Returns:
            torch.Tensor: Predicted scores
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        """
        Full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction: Interaction batch

        Returns:
            torch.Tensor: Predicted scores for all candidate items
        """
        raise NotImplementedError

    def __str__(self):
        """Model prints with number of trainable parameters."""
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """
    Abstract general recommender. All general models should implement this class.
    Provides basic dataset and parameters information.
    """

    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'