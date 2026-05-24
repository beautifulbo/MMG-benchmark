# coding: utf-8

"""
WandB integration utilities for MMG-benchmark.
Provides training and evaluation monitoring with Weights & Biases.
"""

import os
import wandb


class WandBMonitor:
    """WandB monitor for MMG-benchmark training and evaluation."""
    
    def __init__(self, config, project_name="MMG-Benchmark", run_name=None, enabled=True):
        """
        Initialize WandB monitor.
        
        Args:
            config: Configuration object containing model/dataset info
            project_name: WandB project name
            run_name: Specific run name (auto-generated if None)
            enabled: Whether to enable wandb logging
        """
        self.enabled = enabled and self._check_wandb_available()
        self.config = config
        self.run = None
        
        if not self.enabled:
            print("[WandB] WandB monitoring disabled (wandb not installed or disabled)")
            return
            
        # Generate run name if not provided
        if run_name is None:
            model_name = config.get('model', 'unknown')
            dataset_name = config.get('dataset', 'unknown')
            missing_ratio = config.get('missing_ratio', 'none')
            missing_type = config.get('missing_modality_type', 'all')
            run_name = f"{model_name}_{dataset_name}_miss{missing_ratio}_{missing_type}"
        
        # Initialize wandb run
        try:
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=self._extract_config_dict(),
                reinit=True,
                settings=wandb.Settings(start_method="thread")
            )
            
            print(f"[WandB] ✓ Monitoring initialized!")
            print(f"       Project: {project_name}")
            print(f"       Run name: {run_name}")
            print(f"       Track at: {self.run.url}")
            
        except Exception as e:
            print(f"[WandB] ✗ Failed to initialize: {e}")
            self.enabled = False
    
    def _check_wandb_available(self):
        """Check if wandb is available."""
        try:
            import wandb
            return True
        except ImportError:
            return False
    
    def _extract_config_dict(self):
        """Extract relevant config for logging."""
        config_dict = {}
        
        # Model info
        config_dict['model'] = self.config.get('model', 'unknown')
        config_dict['dataset'] = self.config.get('dataset', 'unknown')
        
        # Training hyperparameters
        for key in ['embedding_size', 'learning_rate', 'epochs', 'batch_size',
                    'train_batch_size', 'eval_batch_weight_decay', 'dropout_rate',
                    'n_ui_layers', 'n_mm_layers', 'knn_k', 'alpha', 'lambda_1', 'lambda_2',
                    'infoNCETemp', 'alignBMTemp', 'alignUITemp', 'num_attention_heads',
                    'graph_hidden_dim']:
            val = self.config.get(key)
            if val is not None:
                config_dict[key] = val
        
        # Missing modality info
        config_dict['missing_modal'] = self.config.get('missing_modal', False)
        config_dict['missing_ratio'] = self.config.get('missing_ratio', 0.0)
        config_dict['missing_modality_type'] = self.config.get('missing_modality_type', 'all')
        
        return config_dict
    
    def log_train_metrics(self, epoch_idx, train_loss, lr=None, elapsed_time=None):
        """
        Log training metrics for an epoch.
        
        Args:
            epoch_idx: Current epoch number
            train_loss: Training loss value or tuple of losses
            lr: Current learning rate (optional)
            elapsed_time: Time taken for this epoch (optional)
        """
        if not self.enabled:
            return
        
        metrics = {'epoch': epoch_idx}
        
        # Handle loss (can be single value or tuple)
        if isinstance(train_loss, tuple):
            for i, loss in enumerate(train_loss):
                metrics[f'train/loss_{i+1}'] = loss
            metrics['train/loss_total'] = sum(train_loss)
        else:
            metrics['train/loss_total'] = train_loss
        
        # Add learning rate
        if lr is not None:
            metrics['train/learning_rate'] = lr
        
        # Add timing
        if elapsed_time is not None:
            metrics['train/epoch_time_sec'] = elapsed_time
        
        wandb.log(metrics, step=epoch_idx + 1)
    
    def log_eval_metrics(self, epoch_idx, valid_result=None, test_result=None, 
                         valid_score=None, prefix='eval'):
        """
        Log evaluation metrics.
        
        Args:
            epoch_idx: Current epoch number
            valid_result: Validation metrics dict
            test_result: Test metrics dict  
            valid_score: Main validation score for early stopping
            prefix: Prefix for metric names ('eval' or 'test')
        """
        if not self.enabled:
            return
        
        metrics = {'epoch': epoch_idx}
        
        # Log validation results
        if valid_result:
            for key, value in valid_result.items():
                if isinstance(value, (int, float)):
                    metrics[f'{prefix}/valid_{key}'] = value
        
        if valid_score is not None:
            metrics[f'{prefix}/valid_score'] = valid_score
        
        # Log test results
        if test_result:
            for key, value in test_result.items():
                if isinstance(value, (int, float)):
                    metrics[f'{prefix}/test_{key}'] = value
        
        wandb.log(metrics, step=epoch_idx + 1)
    
    def log_best_metrics(self, best_valid_result, best_test_result, best_epoch):
        """Log final best metrics."""
        if not self.enabled:
            return
        
        summary = {
            'best/epoch': best_epoch,
        }
        
        if best_valid_result:
            for key, value in best_valid_result.items():
                if isinstance(value, (int, float)):
                    summary[f'best/valid_{key}'] = value
        
        if best_test_result:
            for key, value in best_test_result.items():
                if isinstance(value, (int, float)):
                    summary[f'best/test_{key}'] = value
        
        # Log to summary (appears in run table)
        wandb.run.summary.update(summary)
        
        # Also log as metrics
        summary['epoch'] = best_epoch
        wandb.log(summary)
    
    def log_model_checkpoint(self, model_path, alias='best'):
        """Save model checkpoint to wandb artifacts."""
        if not self.enabled or not os.path.exists(model_path):
            return
        
        artifact = wandb.Artifact(f'model-{alias}', type='model')
        artifact.add_file(model_path, 'best_model.pth')
        wandb.log_artifact(artifact)
        print(f"[WandB] Model checkpoint saved as artifact: {alias}")
    
    def watch_model(self, model, log_freq=1000):
        """Watch model gradients and parameters."""
        if not self.enabled:
            return
        
        wandb.watch(model, log='all', log_freq=log_freq)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run:
            wandb.finish()
            print("[WandB] Monitoring finished")


def init_wandb_monitor(config, project_name="MMG-Benchmark", run_name=None, enabled=True):
    """
    Convenience function to create a WandB monitor.
    
    Args:
        config: Configuration object
        project_name: WandB project name
        run_name: Optional specific run name
        enabled: Whether to enable monitoring
        
    Returns:
        WandBMonitor instance
    """
    return WandBMonitor(config, project_name, run_name, enabled)
