# coding: utf-8

"""
Eval script for MMG-benchmark.
Usage:
    python eval.py --dataset baby --checkpoint saved/DGMRec/best_model.pth
"""
import os
import argparse
import torch

from utils.configurator import Config
from data.dataset import RecDataset
from data.dataloader import EvalDataLoader
from models.registry import ModelRegistry
from tasks.link_prediction import LinkPredictionTask
from utils import dict2str


def evaluate(config, model, test_data):
    """Evaluate a trained model."""
    task = LinkPredictionTask(config, model)
    model.eval()

    with torch.no_grad():
        result = task.evaluate(test_data, is_test=True)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DGMRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='gpu_id')
    args = parser.parse_args()

    config_dict = {
        'gpu_id': args.gpu_id,
        'missing_modal': 1,
        'missing_ratio': 0.666
    }

    config = Config(args.model, args.dataset, config_dict)

    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()

    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])

    model = ModelRegistry.get(args.model)(config, train_data).to(config['device'])
    model.load_state_dict(torch.load(args.checkpoint, map_location=config['device']))
    model.eval()

    print("Evaluating on test set...")
    test_result = evaluate(config, model, test_data)
    print("Test Results:")
    print(dict2str(test_result))

    print("\nEvaluating on validation set...")
    valid_result = evaluate(config, model, valid_data)
    print("Validation Results:")
    print(dict2str(valid_result))


if __name__ == '__main__':
    main()