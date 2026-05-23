# coding: utf-8

"""
Evaluation script for MMG-benchmark.
Usage:
    python eval.py --model DGMRec --dataset baby --checkpoint saved/DGMRec/best_model.pth
"""
import os
import argparse

import torch
from logging import getLogger

from utils.configurator import Config
from utils.logger import init_logger
from data.dataset import RecDataset
from data.dataloader import EvalDataLoader
from models.registry import ModelRegistry
from tasks.link_prediction import LinkPredictionTask


def get_model(model_name):
    """Get model class by name."""
    return ModelRegistry.get(model_name)


def quick_start(model, dataset, config_dict, checkpoint_path=None, gpu_id=0):
    """Main entry point for evaluation with optional WandB monitoring."""
    from utils.wandb_monitor import init_wandb_monitor
    
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    
    # Initialize WandB for evaluation (optional)
    wandb_enabled = config_dict.get('wandb_enabled', False)  # Disabled by default for eval
    wandb_project = config_dict.get('wandb_project', 'MMG-Benchmark')
    wandb_run_name = f"eval_{model}_{dataset}"
    
    if wandb_enabled:
        from utils.wandb_monitor import WandBMonitor
        wandb_monitor = WandBMonitor(
            config=config,
            project_name=wandb_project,
            run_name=wandb_run_name,
            enabled=True
        )
    else:
        wandb_monitor = None

    logger.info('██Server: \t' + os.uname().nodename)
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    dataset = RecDataset(config)
    logger.info(str(dataset))

    _, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    train_dataset = dataset  # For additional dataset in eval dataloader
    valid_data = EvalDataLoader(
        config, valid_dataset,
        additional_dataset=train_dataset,
        batch_size=config['eval_batch_size']
    )
    test_data = EvalDataLoader(
        config, test_dataset,
        additional_dataset=train_dataset,
        batch_size=config['eval_batch_size']
    )

    model_instance = get_model(config['model'])(config, train_dataset).to(gpu_id)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=f'cuda:{gpu_id}')
        model_instance.load_state_dict(state_dict)
        logger.info(f'\n✓ Loaded checkpoint: {checkpoint_path}')
    else:
        logger.warning(f'\n⚠ No checkpoint loaded (path not found or not provided): {checkpoint_path}')

    task = LinkPredictionTask(config, model_instance)

    logger.info('\n\n=================================\n\n')

    logger.info('========Evaluating {} on validation set...========='.format(config['model']))
    valid_score, valid_result = task.evaluate(valid_data)
    logger.info('Validation Results:')
    logger.info(dict2str(valid_result))

    logger.info('========Evaluating {} on test set...========='.format(config['model']))
    test_score, test_result = task.evaluate(test_data)
    logger.info('Test Results:')
    logger.info(dict2str(test_result))

    # Log to wandb if enabled
    if wandb_monitor and wandb_monitor.enabled:
        wandb_monitor.log_eval_metrics(
            epoch_idx=0,
            valid_result=valid_result,
            test_result=test_result,
            valid_score=valid_score,
            prefix='final'
        )
        
        wandb_monitor.log_best_metrics(
            best_valid_result=valid_result,
            best_test_result=test_result,
            best_epoch=0
        )
        
        wandb_monitor.finish()

    return {
        'valid_score': valid_score,
        'valid_result': valid_result,
        'test_score': test_score,
        'test_result': test_result
    }