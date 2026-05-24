# coding: utf-8

"""
Node Classification Task for MMG-benchmark
支持 Mario 两阶段流程:
- Stage 1: 用 MarioFeatureExtractor 提取节点表示 -> MLP 分类头
- Stage 2: 用 MAPR + LLM 生成式分类
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from logging import getLogger

from .base_task import BaseTask


class NodeClassificationTask(BaseTask):
    """
    节点分类任务头，支持 Mario 两阶段评估

    Metrics: Accuracy, Macro-F1, Micro-F1
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.metrics = config.get('nc_metrics', ['accuracy', 'macro_f1', 'micro_f1'])
        self.num_classes = config.get('num_classes', None)
        self.label_col = config.get('label_col', 'label')
        self.caption_col = config.get('caption_col', 'caption')
        self.id_col = config.get('id_col', 'id')
        self.use_stage2 = config.get('use_stage2', False)
        self.task_type = config.get('task_type', 'nc')
        self.logger = getLogger()

        self._load_labels(config)
        self._init_stage1_classifier()
        self._check_args()

    def _load_labels(self, config):
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        csv_candidates = [
            os.path.join(dataset_path, f"{config['dataset']}_labels.csv"),
            os.path.join(dataset_path, "node_labels.csv"),
            os.path.join(dataset_path, f"{config['dataset']}.csv"),
        ]
        self.labels = None
        self.label_map = {}
        self.node_ids = None

        for csv_path in csv_candidates:
            if os.path.exists(csv_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    if self.id_col in df.columns and self.label_col in df.columns:
                        self.node_ids = df[self.id_col].values.tolist()
                        raw_labels = df[self.label_col].astype(str).values.tolist()
                        unique_labels = sorted(set(raw_labels))
                        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
                        self.labels = [self.label_map[l] for l in raw_labels]
                        self.num_classes = len(unique_labels) if self.num_classes is None else self.num_classes
                        self.logger.info(f"[NC Task] Loaded {len(self.labels)} labels from {csv_path}, {self.num_classes} classes")
                        break
                except Exception as e:
                    self.logger.warning(f"[NC Task] Failed to load {csv_path}: {e}")

        if self.labels is None:
            self.logger.warning("[NC Task] No label file found, using dummy labels for testing")
            n_nodes = getattr(self.model, 'n_items', 1000)
            self.node_ids = list(range(n_nodes))
            self.labels = np.random.randint(0, max(self.num_classes or 5, 1), n_nodes).tolist()
            self.num_classes = self.num_classes or max(self.labels) + 1
            self.label_map = {str(i): i for i in range(self.num_classes)}

    def _init_stage1_classifier(self):
        actual_dim = self._get_embed_dim()
        self.classifier = nn.Sequential(
            nn.Linear(actual_dim, actual_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(actual_dim // 2, self.num_classes)
        ).to(self.device)

        if hasattr(self.model, 'nc_classifier'):
            self.classifier = self.model.nc_classifier

        self.nc_criterion = nn.CrossEntropyLoss()

    def _get_embed_dim(self):
        if hasattr(self.model, '_get_actual_embed_dim'):
            return self.model._get_actual_embed_dim()
        elif hasattr(self.model, 'embedding_dim'):
            return self.model.embedding_dim
        else:
            return 256

    def _check_args(self):
        if isinstance(self.metrics, str):
            self.metrics = [self.metrics]
        valid_metrics = {'accuracy', 'macro_f1', 'micro_f1', 'precision', 'recall'}
        for m in self.metrics:
            if m.lower() not in valid_metrics:
                raise ValueError(f"Unknown NC metric: {m}. Valid: {valid_metrics}")
        self.metrics = [m.lower() for m in self.metrics]

    def train_epoch(self, train_data, epoch_idx, optimizer, loss_func=None):
        """Stage 1 训练: 特征提取 + MLP 分类"""
        self.model.train()
        self.classifier.train()
        total_loss = 0.0
        loss_batches = []
        all_preds = []
        all_labels_list = []

        for batch_idx, interaction in enumerate(train_data):
            optimizer.zero_grad()

            users, pos_items, neg_items = interaction
            node_ids = torch.unique(torch.cat([pos_items, neg_items]))

            try:
                embeddings = self.model.get_node_embeddings(node_ids)
                logits = self.classifier(embeddings)

                batch_labels = self._get_labels_for_nodes(node_ids.cpu())
                if batch_labels is None:
                    continue
                batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=self.device)

                loss = self.nc_criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loss_batches.append(loss.detach())

                preds = logits.argmax(dim=-1).cpu()
                all_preds.extend(preds.tolist())
                all_labels_list.extend(batch_labels.cpu().tolist())

            except Exception as e:
                self.logger.warning(f"[NC Task] Error at batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(len(loss_batches), 1)
        epoch_acc = self._compute_accuracy(all_preds, all_labels_list)
        self.logger.info(f"[NC Epoch {epoch_idx}] Loss: {avg_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        return avg_loss, loss_batches

    def evaluate(self, eval_data, is_test=False):
        """评估节点分类性能"""
        if self.use_stage2:
            return self._evaluate_stage2(eval_data, is_test)
        else:
            return self._evaluate_stage1(eval_data, is_test)

    def _evaluate_stage1(self, eval_data, is_test=False):
        """Stage 1 评估: 嵌入 + 分类头"""
        self.model.eval()
        self.classifier.eval()

        all_preds = []
        all_labels_list = []

        with torch.no_grad():
            for batch_idx, batched_data in enumerate(eval_data):
                node_ids = batched_data[0] if hasattr(batched_data, '__iter__') else batched_data
                if torch.is_tensor(node_ids):
                    node_ids = node_ids.cpu()

                try:
                    embeddings = self.model.get_node_embeddings(node_ids)
                    logits = self.classifier(embeddings)
                    preds = logits.argmax(dim=-1).cpu().tolist()
                    batch_labels = self._get_labels_for_nodes(node_ids)

                    if batch_labels is not None:
                        all_preds.extend(preds[:len(batch_labels)])
                        all_labels_list.extend(batch_labels)
                except Exception as e:
                    self.logger.warning(f"[NC Eval] Error at batch {batch_idx}: {e}")
                    continue

        metric_dict = self._compute_metrics(all_preds, all_labels_list)
        mode_str = "Test" if is_test else "Valid"
        self.logger.info(f"[{mode_str}] Node Classification Results: {metric_dict}")

        return metric_dict

    def _evaluate_stage2(self, eval_data, is_test=False):
        """Stage 2 评估: MAPR hard routing -> LLM 生成"""
        if not hasattr(self.model, 'stage2_router') or self.model.stage2_router is None:
            self.logger.warning("[NC Task] Stage 2 not initialized, falling back to Stage 1")
            return self._evaluate_stage1(eval_data, is_test)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from .mario_stage2 import (
            MarioPromptTemplate, build_prompt_inputs,
            NC_TASK, LP_TASK, compute_llm_loss
        )

        llm_path = self.config.get('llm_path', 'meta-llama/Llama-3.1-8B')

        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            self.logger.error(f"[NC Task] Cannot load LLM tokenizer from {llm_path}")
            return self._evaluate_stage1(eval_data, is_test)

        self.model.eval()
        self.model.stage2_router.eval()
        self.model.stage2_projector.eval()

        task_desc = NC_TASK if self.task_type == 'nc' else LP_TASK
        modal_names = ["text", "image", "both"]
        all_preds = []
        all_labels_list = []
        modal_counts = {"text": 0, "image": 0, "both": 0}

        text_feats = self.model.t_feat.cpu() if self.model.t_feat is not None else None
        image_feats = self.model.v_feat.cpu() if self.model.v_feat is not None else None

        if text_feats is None or image_feats is None:
            self.logger.warning("[NC Task] Missing features for Stage 2, falling back to Stage 1")
            return self._evaluate_stage1(eval_data, is_test)

        with torch.no_grad():
            for batch_idx, batched_data in enumerate(tqdm(eval_data, desc="Stage2 Evaluation")):
                node_ids = batched_data[0] if hasattr(batched_data, '__iter__') else batched_data
                if torch.is_tensor(node_ids):
                    node_ids = node_ids.cpu()

                for node_id in node_ids:
                    node_id_val = node_id.item() if torch.is_tensor(node_id) else int(node_id)
                    gt_label = self._get_label_for_node(node_id_val)
                    if gt_label is None:
                        continue

                    gt_label_name = [k for k, v in self.label_map.items() if v == gt_label]
                    gt_label_str = gt_label_name[0] if gt_label_name else str(gt_label)

                    h_text = text_feats[node_id_val].to(self.device)
                    h_image = image_feats[node_id_val].to(self.device)

                    hop1 = self._get_hop_neighbors(node_id_val, hop=1)
                    hop2 = self._get_hop_neighbors(node_id_val, hop=2)
                    degree = len(hop1)

                    z_v = self.model.stage2_router.build_router_input(
                        h_text, h_image,
                        torch.tensor(hop1, dtype=torch.long, device=self.device),
                        torch.tensor(hop2, dtype=torch.long, device=self.device),
                        degree,
                        all_h_text=text_feats.to(self.device),
                        all_h_image=image_feats.to(self.device),
                    )
                    probs, _ = self.model.stage2_router(z_v.unsqueeze(0))
                    chosen_modal = modal_names[probs.argmax(dim=-1).item()]
                    modal_counts[chosen_modal] += 1

                    caption = self._get_caption_for_node(node_id_val)
                    template = MarioPromptTemplate(
                        text_feats, image_feats, node_id_val,
                        task_description=task_desc,
                        raw_caption=caption,
                        hop1_neighbors=torch.tensor(hop1, dtype=torch.long),
                        hop2_neighbors=torch.tensor(hop2, dtype=torch.long),
                    )
                    special_tokens = template.generate_template(modal=chosen_modal)
                    projected_tokens = self.model.stage2_projector(special_tokens)

                    prompt_str = f"{task_desc}\n\nNode content: {caption}\n\nGraph context tokens: "
                    prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=True).input_ids.to(
                        self.device)
                    prompt_embeds = self.model.llm_embed_layer(prompt_ids) if hasattr(self.model,
                                                                                   'llm_embed_layer') else None

                    if prompt_embeds is None:
                        pred_label = self._stage2_fallback_predict(node_id_val, chosen_modal)
                    else:
                        gen_embeds = torch.cat([prompt_embeds, projected_tokens.unsqueeze(0)], dim=1)
                        output_ids = self.model.stage2_llm.generate(
                            inputs_embeds=gen_embeds,
                            max_new_tokens=20,
                            do_sample=False,
                        ) if hasattr(self.model, 'stage2_llm') and self.model.stage2_llm is not None else None

                        if output_ids is not None:
                            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                            pred_label = self._match_prediction_to_label(pred_text)
                        else:
                            pred_label = self._stage2_fallback_predict(node_id_val, chosen_modal)

                    all_preds.append(pred_label)
                    all_labels_list.append(gt_label)

        self.logger.info(f"[NC Stage2] Modal distribution: {modal_counts}")
        metric_dict = self._compute_metrics(all_preds, all_labels_list)
        mode_str = "Test" if is_test else "Valid"
        self.logger.info(f"[{mode_str} Stage2] Results: {metric_dict}")

        return metric_dict

    def _stage2_fallback_predict(self, node_id, modal):
        embeddings = self.model.get_node_embeddings(torch.tensor([node_id]).to(self.device))
        logits = self.classifier(embeddings)
        return logits.argmax(dim=-1).item()

    def _match_prediction_to_label(self, pred_text):
        best_match = 0
        best_score = 0
        pred_lower = pred_text.lower().strip()
        for label_name, label_idx in self.label_map.items():
            if str(label_idx) == pred_text.strip() or label_name.lower() == pred_lower:
                return label_idx
            elif label_name.lower() in pred_lower or pred_lower in label_name.lower():
                best_match = label_idx
                best_score = 0.5
        return best_match if best_score > 0 else 0

    def _get_labels_for_nodes(self, node_ids):
        if self.labels is None or self.node_ids is None:
            return None
        labels = []
        for nid in node_ids:
            nid_val = nid.item() if torch.is_tensor(nid) else int(nid)
            idx = self.node_ids.index(nid_val) if nid_val in self.node_ids else -1
            if idx >= 0:
                labels.append(self.labels[idx])
            else:
                labels.append(0)
        return labels

    def _get_label_for_node(self, node_id):
        if self.labels is None or self.node_ids is None:
            return None
        idx = self.node_ids.index(node_id) if node_id in self.node_ids else -1
        return self.labels[idx] if idx >= 0 else None

    def _get_caption_for_node(self, node_id):
        if not hasattr(self, 'captions') or self.captions is None:
            return f"Node {node_id}"
        return self.captions.get(node_id, f"Node {node_id}")

    def _get_hop_neighbors(self, node_id, hop=1):
        if hasattr(self.model, 'graph'):
            g = self.model.graph
            try:
                succ = g.successors(node_id) if hop == 1 else []
                if hop == 2:
                    hop2_set = set()
                    for n in g.successors(node_id):
                        n_item = n.item() if isinstance(n, torch.Tensor) else n
                        for h in g.successors(n_item):
                            h_item = h.item() if isinstance(h, torch.Tensor) else h
                            if h_item != node_id:
                                hop2_set.add(h_item)
                    succ = list(hop2_set)
                return [s.item() if isinstance(s, torch.Tensor) else s for s in succ][:self.config.get('k_neighbors', 5)]
            except:
                return []
        return []

    def _compute_metrics(self, predictions, labels):
        if len(predictions) == 0 or len(labels) == 0:
            return {f'{m}@1': 0.0 for m in self.metrics}

        predictions = np.array(predictions)
        labels = np.array(labels)

        metric_dict = {}

        if 'accuracy' in self.metrics:
            metric_dict['accuracy@1'] = round(float((predictions == labels).mean()), 4)

        if 'macro_f1' in self.metrics or 'micro_f1' in self.metrics or 'precision' in self.metrics or 'recall' in self.metrics:
            try:
                from sklearn.metrics import f1_score, precision_score, recall_score
                if 'macro_f1' in self.metrics:
                    metric_dict['macro_f1@1'] = round(float(f1_score(labels, predictions, average='macro', zero_division=0)), 4)
                if 'micro_f1' in self.metrics:
                    metric_dict['micro_f1@1'] = round(float(f1_score(labels, predictions, average='micro', zero_division=0)), 4)
                if 'precision' in self.metrics:
                    metric_dict['precision@1'] = round(float(precision_score(labels, predictions, average='weighted', zero_division=0)), 4)
                if 'recall' in self.metrics:
                    metric_dict['recall@1'] = round(float(recall_score(labels, predictions, average='weighted', zero_division=0)), 4)
            except ImportError:
                self.logger.warning("sklearn not available, skipping F1/Precision/Recall metrics")

        return metric_dict

    def _compute_accuracy(self, predictions, labels):
        if len(predictions) == 0:
            return 0.0
        return float((np.array(predictions) == np.array(labels)).mean())

    def predict(self, interaction):
        node_ids = interaction[0] if hasattr(interaction, '__iter__') else interaction
        embeddings = self.model.get_node_embeddings(node_ids)
        logits = self.classifier(embeddings)
        return logits
