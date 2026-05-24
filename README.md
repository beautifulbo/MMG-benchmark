# MMG-benchmark

A modular benchmark for evaluating multimodal graph models in missing modality scenarios.

## Overview

MMG-benchmark provides a unified framework for training and evaluating multimodal graph models on scenarios including missing modalities and new items. It is designed for researchers to easily benchmark different models on the same evaluation pipeline.

## Supported Models

| Model | Description | Stage 1 (Feature) | Stage 2 (LLM) | Tasks |
|-------|-------------|:-----------------:|:--------------:|:-----:|
| **DGMRec** | Disentangling and Generating Modalities for Recommendation | ✅ GNN-based | ❌ | LP |
| **CRLMMNAR** | Contrastive Representation Learning for Missing Modality | ✅ GNN-based | ❌ | LP |
| **Mario** | Multimodal Graph Learning with Distance-aware Attention | ✅ Dist-aware Attn | ✅ MAPR + LoRA | LP, NC |

## Project Structure

```
MMG-benchmark/
├── configs/              # YAML configuration files
│   ├── dataset/         # Dataset-specific settings
│   │   └── baby.yaml    # Dataset config example
│   ├── model/           # Model hyperparameters
│   │   ├── DGMRec.yaml  # DGMRec model config
│   │   ├── CRLMMNAR.yaml
│   │   └── Mario.yaml   # Mario model config (Stage1 + Stage2)
│   └── overall.yaml     # General settings
├── data/                # Data preprocessing module
│   ├── dataloader.py     # TrainDataLoader, EvalDataLoader
│   ├── dataset.py       # RecDataset for loading interactions
│   ├── preprocessing/   # Data preprocessing scripts
│   │   └── missing_modality.py  # Missing modality generation
│   └── utils/           # Graph and data utilities
├── models/              # Model framework module
│   ├── base/            # Base classes (AbstractRecommender, Loss, MI estimator)
│   │   ├── abstract_recommender.py
│   │   ├── loss.py      # BPR, MSE, Emb, L2, Dice losses
│   │   └── mi_estimator.py  # CLUB sample estimator
│   ├── dgmrec.py        # DGMRec model implementation
│   ├── crlmmnar.py      # CRLMMNAR model implementation
│   ├── mario.py         # Mario model (Stage 1: Feature Extractor)
│   ├── mario_stage2.py  # Mario Stage 2 components (MAPR, Loss, Prompt)
│   └── registry.py      # Model registry for plugin-style registration
├── tasks/               # Task head module
│   ├── base_task.py     # BaseTask abstract class
│   ├── link_prediction.py  # Link prediction / recommendation task (Top-K metrics)
│   ├── node_classification.py  # Node classification task (Accuracy, F1) ✅ Implemented
│   └── metrics.py       # Evaluation metrics computation
├── utils/
│   ├── configurator.py  # YAML config loading with priority override
│   ├── logger.py        # Logging utilities
│   └── graph_utils_mario.py  # Graph distance matrix computation for Mario
├── train.py             # Training entry point (supports --task nc/lp)
├── train_stage2.py      # Stage 2 LLM fine-tuning entry (Mario only)
└── eval.py              # Evaluation entry point
```

## Installation

```bash
conda create -n mmg-benchmark python=3.9
conda activate mmg-benchmark
pip install -r requirements.txt
```

Required dependencies (see `requirements.txt`):

### Core Dependencies
- PyTorch >= 2.0
- torch-geometric / DGL (for graph operations)
- numpy, pandas, scipy
- scikit-learn, pyyaml

### Mario Model Additional Dependencies
```bash
# For Stage 1 (graph distance matrix + attention)
pip install dgl>=1.0.0

# For Stage 2 (LLM fine-tuning)
pip install transformers>=4.36.0
pip install peft>=0.7.0        # LoRA fine-tuning
pip install accelerate>=0.25.0  # Device mapping
# Optional: networkx (for distance matrix computation)
pip install networkx>=3.0
```

## Data Preparation

### Download Datasets

Download datasets (Baby/Sports/Clothing/Reddit) from MMRec Google Drive:

1. Visit the [MMRec Google Drive](https://github.com/enoche/MMRec) repository
2. Navigate to the Google Drive link in their README
3. Download the following datasets:
   - `baby.tar`
   - `sports.tar`
   - `clothing.tar`

Alternatively, direct Google Drive link: [MMRec Dataset](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)

### Extract Data

After downloading, extract the datasets:

```bash
mkdir -p data
tar -xf baby.tar -C data/
tar -xf sports.tar -C data/
tar -xf clothing.tar -C data/
```

Expected directory structure after extraction:

```
data/
├── baby/
│   ├── baby.inter           # User-item interactions
│   ├── image_feat.npy       # Visual features (CNN/ViT extracted)
│   ├── text_feat.npy        # Text features (BERT/LLM extracted)
│   ├── u_id_mapping.csv     # User ID mapping
│   ├── i_id_mapping.csv     # Item ID mapping
│   └── missing_items_0.666.npy  # Missing modality masks (generated)
├── sports/
│   └── ...
├── clothing/
│   └── ...
└── reddit/                   # For node classification tasks
    ├── reddit.inter
    ├── image_feat.npy
    ├── text_feat.npy
    ├── reddit_labels.csv    # Node labels (id, label, caption columns)
    └── ...
```

### Dataset Format

Each dataset should contain:
- `{name}.inter` - Interaction TSV with columns: `userID`, `itemID`, `rating`, `timestamp`, `x_label`
  - `x_label`: 0 = training, 1 = validation, 2 = test
- `image_feat.npy` - Pre-extracted CNN image features (NumPy array)
- `text_feat.npy` - Pre-extracted text features (NumPy array)
- `{name}_labels.csv` (for NC task): Columns `id`, `label`, `caption`
- `u_id_mapping.csv` - User ID mapping
- `i_id_mapping.csv` - Item ID mapping

### Preprocessing for Missing Modality

```bash
cd MMG-benchmark
python -m data.preprocessing.missing_modality --dataset baby
```

This generates `missing_items_{ratio}.npy` in the dataset directory.

## Quick Start

### Missing Modality Preprocessing

```bash
cd MMG-benchmark
python -m data.preprocessing.missing_modality --dataset baby
```

---

### 🚀 Training DGMRec / CRLMMNAR (Link Prediction)

```bash
python train.py --model DGMRec --dataset baby --missing_modal 1 --missing_ratio 0.666 --task lp
```

Arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `DGMRec` | Model name (`DGMRec`, `CRLMMNAR`, `Mario`) |
| `--dataset` | `baby` | Dataset name |
| `--missing_modal` | `1` | Enable missing modality setting |
| `--missing_ratio` | `0.666` | Missing ratio (0.1 ~ 0.9) |
| `--missing_modality_type` | `all` | Which modality: `all`, `text`, `image` |
| `--task` / `-t` | `lp` | Task type: `lp` (link prediction), `nc` (node classification) |
| `--gpu_id` | `0` | GPU device ID |

---

### 🎮 Training Mario Model

#### Stage 1: Feature Extraction (Link Prediction)

```bash
python train.py --model Mario --dataset baby --missing_modal 1 --missing_ratio 0.3 --task lp
```

#### Stage 1: Feature Extraction (Node Classification)

```bash
python train.py --model Mario --dataset reddit --task nc
```

Mario-specific config options in `configs/model/Mario.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_size` | `3584` | Feature dimension (match your pre-extracted features) |
| `num_heads` | `8` | Number of attention heads |
| `buckets_num` | `6` | Distance bucket count for position encoding |
| `n_layers` | `6` | Number of stacked attention layers |
| `temperature` | `0.07` | InfoNCE loss temperature |
| `k_neighbors` | `5` | Top-K neighbors for prompt template |

---

### 🤖 Stage 2: LLM Fine-tuning (Mario Only)

Stage 2 trains a **MAPR router** (modality-adaptive routing) + **LoRA-adapted LLM** on top of frozen Stage 1 features.

```bash
python train_stage2.py \
    --model Mario \
    --dataset reddit \
    --stage1_checkpoint saved/Mario-/best_model.pth \
    --llm_path meta-llama/Llama-3.1-8B \
    --task nc \
    --num_epochs 10 \
    --batch_size 8
```

Stage 2 specific arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage1_checkpoint` | *(required)* | Path to trained Stage 1 checkpoint |
| `--llm_path` | `meta-llama/Llama-3.1-8B` | HuggingFace LLM path |
| `--llm_hidden_dim` | `4096` | LLM hidden dimension (for projector) |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout rate |
| `--router_hidden_dim` | `256` | MAPR router hidden dimension |
| `--kl_weight` | `0.1` | KL regularization weight |
| `--save_dir` | `saved/mario_stage2` | Checkpoint save directory |

**Supported LLMs**: Any HuggingFace causal LM (Llama-3, Qwen2-VL, Mistral, etc.)

---

### 📊 Evaluation

#### Evaluate Stage 1 Models (Link Prediction)

```bash
python eval.py --model Mario --dataset baby --checkpoint saved/Mario-/best_model.pth --gpu_id 0
```

#### Evaluate Stage 1 Models (Node Classification)

```bash
python eval.py --model Mario --dataset reddit --checkpoint saved/Mario-/best_model.pth --task nc
```

#### Evaluate Stage 2 Models

Stage 2 evaluation is integrated into `train_stage2.py` (automatic after training). The evaluation uses **MAPR hard routing** → selects optimal modality per node → LLM generates prediction.

---

## Supported Tasks & Metrics

### Link Prediction (LP)

| Metric | Description |
|--------|-------------|
| Recall@K | Recall at top-K recommendations |
| NDCG@K | Normalized Discounted Cumulative Gain |
| Precision@K | Precision at top-K |
| MAP@K | Mean Average Precision |

### Node Classification (NC) ✅ Fully Implemented

| Metric | Description |
|--------|-------------|
| Accuracy@1 | Classification accuracy |
| Macro-F1@1 | Macro-averaged F1 score |
| Micro-F1@1 | Micro-averaged F1 score |
| Precision@1 | Weighted precision |
| Recall@1 | Weighted recall |

---

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
        # Return training loss (scalar tensor)
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        # Return scores: (batch_size, n_items)
        raise NotImplementedError
```

Then register in `models/__init__.py`:

```python
from .my_model import MyModel
ModelRegistry._models['MyModel'] = MyModel
```

Train with:

```bash
python train.py --model MyModel --dataset baby
```

## Mario Architecture Details

### Stage 1: Distance-Aware Attention Feature Extractor

```
Input: Text Features (N, d) + Image Features (N, d) + Global Distance Matrix (N, N)
         ↓
    ┌─────────────────────────────────────┐
    │   MultiHeadAttention (per layer)    │
    │   ├─ Q/K/V Linear Projections       │
    │   ├─ Scaled Dot-Product Attention    │
    │   ├─ ★ Distance-aware Position Bias  │ ← Learnable buckets over shortest paths
    │   └─ Output: Update center token only│
    │         (stacked × n_layers)          │
    └─────────────────────────────────────┘
         ↓
    Dual Channel: [Text Encoder] + [Image Encoder] (independent parameters)
         ↓
Output: Enhanced text embeddings, Enhanced image embeddings
Loss: InfoNCE(text_emb, image_emb) — cross-modal contrastive learning
```

**Key Innovation**: Position encoding based on **graph shortest path distances** using learnable bucket embeddings, enabling the model to capture structural relationships between nodes.

### Stage 2: Modality-Adaptive Instruction Tuning

```
Input: Frozen Stage 1 Features + Graph Structure + Node Labels/Captions
         ↓
    ┌──────────────────────────────────────────┐
    │  MAPRouter: z_v = [h_t; h_v; φ¹; φ²; log(d)] │
    │       ↓ MLP + Softmax                    │
    │  p_v = [p_text, p_image, p_both] ← Routing probs │
    └──────────────────────────────────────────┘
         ↓
    ┌──────────────────────────────────────────┐
    │  Prompt Template Bank (selected by MAPR) │
    │  ├─ text:   {<GT_center>, <GT_n1>, ...}  │
    │  ├─ image:  {<GI_center>, <GI_n1>, ...}  │
    │  └─ both:   {<GT>, <GI>, ..., <GT>, <GI>}│
    │  (via SpecialTokenProjector: d → llm_dim) │
    └──────────────────────────────────────────┘
         ↓
    ┌──────────────────────────────────────────┐
    │  LLM (Llama-3.1-8B) + LoRA              │
    │  Input: [Task Desc; Caption; Graph Tokens; Answer] │
    │  Output: Predicted label/classification   │
    └──────────────────────────────────────────┘

Loss: L_S2 = Σ_v [ Σ_k q_v^(k) · ℓ_v^(k) + λ·KL(q‖p) ]
  where q = softmax(-losses) (performance posterior), p = router output
```

## Configuration

Configs are loaded with priority: **command line > config_dict > YAML files**

### Key Config Files

| File | Purpose |
|------|---------|
| `configs/overall.yaml` | General settings (epochs, batch_size, metrics, device) |
| `configs/dataset/{name}.yaml` | Dataset paths, feature files, field mappings |
| `configs/model/{name}.yaml` | Model hyperparameters (architecture, training, Stage 2) |

### Example Config Override

```python
config_dict = {
    'learning_rate': 0.001,
    'epochs': 500,
    'missing_modal': 1,
    'task_type': 'nc',          # or 'lp'
    'use_stage2': False,        # Enable Stage 2 LLM fine-tuning
}
config = Config('Mario', 'reddit', config_dict)
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'dgl'` | Install DGL: `pip install dgl` |
| `CUDA out of memory` (Stage 2) | Reduce `batch_size`, use smaller LLM (e.g., Llama-3.1-8B-Instruct), enable CPU offload |
| Distance matrix too large | Set `buckets_num=4` to reduce granularity, or limit graph size |
| `sklearn not available` (NC metrics) | `pip install scikit-learn` |
| Feature dimension mismatch | Check `embedding_size` matches your `.npy` feature files |

### Performance Tips

1. **Distance Matrix**: Computed once and cached to `saved/mario_dist_matrix.pt`. Delete to recompute.
2. **Large Graphs (>10K nodes)**: Uses approximate BFS for distance matrix instead of exact NetworkX.
3. **Stage 2 Memory**: Use `lora_r=8` and `batch_size=4` for GPUs < 24GB VRAM.
4. **Multi-GPU**: Stage 2 supports `accelerate` device_map="auto" for model parallelism.

## License

MIT License
