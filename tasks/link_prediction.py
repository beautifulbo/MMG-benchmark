# coding: utf-8

"""
Link prediction task head for recommendation scenarios.
Supports top-k ranking evaluation with metrics like Recall, NDCG, Precision, MAP.
"""
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from logging import getLogger

from .base_task import BaseTask
from .metrics import metrics_dict


# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Recall2', 'Precision', 'NDCG', 'MAP']}


class LinkPredictionTask(BaseTask):
    """
    Task head for link prediction / recommendation.

    Metrics supported: Recall, Recall2, Precision, NDCG, MAP
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.save_recom_result = config['save_recommended_topk']
        self._check_args()
        self.logger = getLogger()

    def train_epoch(self, train_data, epoch_idx, optimizer, loss_func):
        """
        Train for one epoch.

        Args:
            train_data: Training data loader
            epoch_idx: Current epoch index
            optimizer: Optimizer
            loss_func: Loss function

        Returns:
            tuple: (total_loss, loss_batches)
        """
        self.model.train()
        total_loss = None
        loss_batches = []

        for batch_idx, interaction in enumerate(train_data):
            optimizer.zero_grad()
            losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()

            if torch.isnan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)

            loss.backward()
            optimizer.step()
            loss_batches.append(loss.detach())

        return total_loss, loss_batches

    def evaluate(self, eval_data, is_test=False):
        """
        Evaluate the model with top-k metrics.

        Args:
            eval_data: Evaluation data loader
            is_test: Whether this is test evaluation

        Returns:
            dict: Evaluation metrics like {'Recall@20': 0.05, 'NDCG@10': 0.03, ...}
        """
        self.model.eval()
        batch_matrix_list = []

        with torch.no_grad():
            for batch_idx, batched_data in enumerate(eval_data):
                scores = self.model.full_sort_predict(batched_data)
                masked_items = batched_data[1]  # pos item
                scores[masked_items[0], masked_items[1]] = -1e10
                _, topk_index = torch.topk(scores, max(self.topk), dim=-1)
                batch_matrix_list.append(topk_index)

        return self._compute_metrics(batch_matrix_list, eval_data, is_test)

    def _compute_metrics(self, batch_matrix_list, eval_data, is_test=False):
        """Compute top-k metrics from batch results."""
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        # Save recommendation results if enabled
        if self.save_recom_result and is_test:
            dataset_name = self.config['dataset']
            model_name = self.config['model']
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config['recommend_topk'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            from ..utils import get_local_time
            file_path = os.path.join(dir_name, '{}-{}-top{}-{}.csv'.format(
                model_name, dataset_name, max_k, get_local_time()))
            import pandas as pd
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, 'id', eval_data.get_eval_users())
            x_df.columns = ['id'] + ['top_' + str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep='\t', index=False)

        assert len(pos_len_list) == len(topk_index)

        # Check if recommended items match positive items
        bool_rec_matrix = []
        for m, n in zip(pos_items, topk_index):
            bool_rec_matrix.append([True if i in m else False for i in n])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        # Compute metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)

        return metric_dict

    def _calculate_metrics(self, pos_len_list, topk_index):
        """Calculate metrics for all users."""
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)

    def _check_args(self):
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError('topk must be a positive integer or list')
        else:
            raise TypeError('The topk must be a integer, list')