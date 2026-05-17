# MMG-benchmark

A modular benchmark for evaluating multimodal graph models in missing modality scenarios.

## Overview

MMG-benchmark provides a unified framework for training and evaluating multimodal graph models on scenarios including missing modalities and new items. It is designed for researchers to easily benchmark different models on the same evaluation pipeline.

## Project Structure

```
MMG-benchmark/
├── configs/              # YAML configuration files
│   ├── dataset/         # Dataset-specific settings
│   ├── model/           # Model hyperparameters
│   └── overall.yaml     # General settings
├── data/                # Data preprocessing module
│   ├── dataloader.py     # TrainDataLoader, EvalDataLoader
│   ├── dataset.py       # RecDataset for loading interactions
│   ├── preprocessing/   # Data preprocessing scripts
│   └── utils/           # Graph and data utilities
├── models/              # Model framework module
│   ├── base/            # Base classes (AbstractRecommender, Loss)
│   ├── dgmrec.py        # DGMRec model implementation
│   └── registry.py      # Model registry for plugin-style registration
├── tasks/               # Task head module
│   ├── base_task.py     # BaseTask abstract class
│   ├── link_prediction.py # Recommendation task (top-k ranking)
│   └── metrics.py       # Evaluation metrics (Recall, NDCG, etc.)
├── utils/
│   ├── configurator.py  # YAML config loading
│   └── logger.py        # Logging utilities
├── train.py             # Training entry point
└── eval.py              # Evaluation entry point
```

## Installation

```bash
conda create -n mmg-benchmark python=3.9
conda activate mmg-benchmark
pip install -r requirements.txt
```

Required dependencies (see `requirements.txt`):

- PyTorch >= 2.0
- torch-geometric
- numpy, pandas, scipy
- scikit-learn, pyyaml

## Data Preparation

1. Download datasets (Baby/Sports/Clothing) from [MMRec](https://github.com/enoche/MMRec) Google Drive
2. Place extracted data in `data/{dataset}/` directory
3. Each dataset should contain:
   - `{name}.inter` - Interaction TSV (userID, itemID, rating, timestamp, x_label)
   - `image_feat.npy` - Pre-extracted image features
   - `text_feat.npy` - Pre-extracted text features

## Quick Start

### Missing Modality Preprocessing

```bash
cd MMG-benchmark
python -m data.preprocessing.missing_modality --dataset baby
```

### Training

```bash
python train.py --dataset baby --missing_items 1 --gpu_id 0
```

Arguments:

- `--dataset`: Dataset name (baby, sports, clothing)
- `--missing_items`: 1 = Missing Modality Setting, 0 = No Missing
- `--missing_ratio`: Missing ratio (default 0.666)
- `--gpu_id`: GPU device ID

### Evaluation

```bash
python eval.py --dataset baby --checkpoint saved/model.pth --gpu_id 0
```

## Adding New Models

MMG-benchmark uses a plugin-style model registry. To add a new model:

```python
from models.registry import register_model
from models.base.abstract_recommender import GeneralRecommender

@register_model('MyModel')
class MyModel(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MyModel, self).__init__(config, dataset)
        # Initialize your model

    def calculate_loss(self, interaction):
        # Return training loss
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        # Return scores for all items
        raise NotImplementedError
```

Then train with:

```bash
python train.py --model MyModel --dataset baby
```

## Extending Tasks

The `tasks/` module provides task heads. Current tasks:

- **LinkPredictionTask**: Top-k recommendation with Recall@K, NDCG@K, etc.
- **NodeClassificationTask**: Placeholder for graph node classification (not yet implemented)

To add a new task, subclass `BaseTask`:

```python
from tasks.base_task import BaseTask

class MyTask(BaseTask):
    def __init__(self, config, model):
        super().__init__(config, model)

    def train_epoch(self, train_data, epoch_idx, optimizer, loss_func):
        # Training logic
        raise NotImplementedError

    def evaluate(self, eval_data, is_test=False):
        # Evaluation logic
        raise NotImplementedError
```

## Configuration

Configs are loaded with priority: **command line > config_dict > YAML files**

### Key Config Files

| File | Purpose |
|------|---------|
| `configs/overall.yaml` | Training/evaluation settings (epochs, batch_size, metrics) |
| `configs/dataset/{name}.yaml` | Dataset paths and features |
| `configs/model/{name}.yaml` | Model hyperparameters |

### Example Config Override

```python
config_dict = {
    'learning_rate': 0.001,
    'epochs': 500,
    'missing_modal': 1
}
config = Config('DGMRec', 'baby', config_dict)
```

## Architecture Notes

### Data Flow

```
Raw Data → RecDataset.split() → TrainDataLoader/EvalDataLoader
                                              ↓
                                          Model
                                              ↓
                                          Task.head
                                              ↓
                                         Metrics
```

### Model Base Classes

- `AbstractRecommender`: Base for all models, defines `calculate_loss()`, `predict()`, `full_sort_predict()`
- `GeneralRecommender`: Extends AbstractRecommender, adds multimodal feature loading (`v_feat`, `t_feat`)

### Loss Components (DGMRec example)

- BPR loss for recommendation
- InfoNCE for modality alignment
- MI estimation (CLUB) for disentanglement
- MSE reconstruction for modality generation
- Alignment losses (UI and BM)


