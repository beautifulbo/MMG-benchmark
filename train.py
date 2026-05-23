# coding: utf-8

"""
Train script for MMG-benchmark.
Usage:
    python train.py --dataset baby --missing_items 1
"""
import os
import argparse
import itertools
import platform
from time import time

import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from logging import getLogger

from utils.configurator import Config
from utils import init_seed, early_stopping, dict2str, get_local_time
from data.dataset import RecDataset
from data.dataloader import TrainDataLoader, EvalDataLoader
from models.registry import ModelRegistry
from models.base.abstract_recommender import GeneralRecommender
from tasks.link_prediction import LinkPredictionTask


def get_model(model_name):
    """Get model class by name."""
    return ModelRegistry.get(model_name)


def get_trainer():
    """Get trainer class."""
    return Trainer


class Trainer:
    """
    Trainer for model training and evaluation.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()

        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']
        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.model.init_mi_estimator()

        lr_scheduler = config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.task = LinkPredictionTask(config, model)

    def _build_optimizer(self):
        """Build optimizer."""
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        """Train for one epoch."""
        if not self.req_training:
            return 0.0, []

        self.model.train()
        self.model.item_image_estimator.eval()
        self.model.item_text_estimator.eval()

        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []

        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
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
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())

        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, type_='val'):
        """Validate for one epoch."""
        valid_result = self.task.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        """Generate training loss output string."""
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, save_dir=None, verbose=True):
        """Train the model."""
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()

            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % (
                    epoch_idx, valid_end_time - valid_start_time, valid_score
                )
                valid_result_output = 'valid result: \n' + dict2str(valid_result)

                _, test_result = self._valid_epoch(test_data, type_='test')
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))

                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


def quick_start(model, dataset, config_dict, save_model=True):
    """Main entry point for training."""
    from utils.logger import init_logger

    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()

    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    dataset = RecDataset(config)
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])

    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    best_test_idx = 0
    idx = 0

    logger.info('\n\n=================================\n\n')

    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    combinators = list(itertools.product(*hyper_ls))
    total_loops = len(combinators)

    for i, hyper_tuple in enumerate(combinators):
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))

        train_data.pretrain_setup()
        model = get_model(config['model'])(config, train_data).to(config['device'])
        model.logger = logger
        logger.info(model)

        trainer = Trainer(config, model)

        save_dir = config['save_name'][:-4] + "-" + str(hyper_tuple) if config.get('save_name') else None
        model.save_dir = save_dir

        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=save_model, save_dir=save_dir
        )
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\nValid: {},\nTest: {}\n\n\n'.format(
            config['hyper_parameters'],
            hyper_ret[best_test_idx][0],
            dict2str(hyper_ret[best_test_idx][1]),
            dict2str(hyper_ret[best_test_idx][2])))

    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(
            config['hyper_parameters'], p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(
        config['hyper_parameters'],
        hyper_ret[best_test_idx][0],
        dict2str(hyper_ret[best_test_idx][1]),
        dict2str(hyper_ret[best_test_idx][2])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DGMRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=str, default='1', help='gpu_id')
    parser.add_argument('--missing_modal', type=int, default=1, help='missing_modal')
    parser.add_argument('--missing_ratio', type=str, default='0.666', help='missing_ratio')
    parser.add_argument('--missing_modality_type', type=str, default='all',
                        choices=['text', 't', 'image', 'v', 'visual', 'all'],
                        help='which modality to make missing (text/image/all)')

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
        'missing_modal': args.missing_modal,
        'missing_ratio': eval(args.missing_ratio),
        'missing_modality_type': args.missing_modality_type
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)