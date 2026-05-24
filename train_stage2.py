# coding: utf-8

"""
Mario Stage 2 Training Script
模态自适应图指令微调训练入口

用法:
    python train_stage2.py --model Mario --dataset reddit \\
        --stage1_checkpoint saved/Mario-/best_model.pth \\
        --llm_path meta-llama/Llama-3.1-8B \\
        --task nc --num_epochs 10

功能:
    - 加载 Stage 1 冻结的 MarioFeatureExtractor
    - 初始化 MAPRouter (模态自适应路由器)
    - 初始化 SpecialTokenProjector (特征投影到 LLM 空间)
    - 加载 LLM + LoRA
    - 使用 Stage2CompositeLoss 联合训练 MAPR + Projector + LoRA
    - 支持 NC (节点分类) 和 LP (链接预测) 任务
"""
import os
import sys
import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import tqdm
from logging import getLogger, StreamHandler, Formatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.configurator import Config
from utils.logger import init_logger
from models.registry import ModelRegistry
from models.mario_stage2 import (
    MAPRouter, SpecialTokenProjector, Stage2CompositeLoss,
    Stage2Dataset, MarioPromptTemplate,
    NC_TASK, LP_TASK, build_prompt_inputs, compute_llm_loss
)


class MarioStage2Trainer:
    """
    Mario Stage 2 训练器: MAPR + LLM LoRA 联合微调
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logger()
        self.modal_names = ["text", "image", "both"]

        self._load_config()
        self._load_stage1_model()
        self._load_dataset()
        self._init_stage2_components()
        self._init_optimizer()

    def _setup_logger(self):
        logger = getLogger("MarioStage2")
        logger.setLevel(20)
        if not logger.handlers:
            handler = StreamHandler()
            handler.setFormatter(Formatter('[%(name)s] %(message)s'))
            logger.addHandler(handler)
        return logger

    def _load_config(self):
        model_name = getattr(self.args, 'model', 'Mario')
        dataset_name = getattr(self.args, 'dataset', 'reddit')
        config_dict = {
            'model': model_name,
            'dataset': dataset_name,
            'task_type': getattr(self.args, 'task', 'nc'),
            'use_stage2': True,
            'gpu_id': getattr(self.args, 'gpu_id', '0'),
        }
        self.config = Config(model_name, dataset_name, config_dict)
        self.logger.info(f"Config loaded: model={model_name}, dataset={dataset_name}, task={config_dict['task_type']}")

    def _load_stage1_model(self):
        checkpoint_path = self.args.stage1_checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

        ModelClass = ModelRegistry.get(self.config['model'])
        if ModelClass is None:
            raise ValueError(f"Model '{self.config['model']}' not registered")

        from data.dataset import RecDataset
        dataset = RecDataset(self.config)
        train_ds, _, _ = dataset.split()

        self.stage1_model = ModelClass(self.config, train_ds).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.stage1_model.load_state_dict(state_dict, strict=False)
        self.stage1_model.eval()

        for param in self.stage1_model.parameters():
            param.requires_grad = False

        self.embed_dim = self.stage1_model._get_actual_embed_dim()
        self.logger.info(f"[Stage2] Stage 1 model loaded from {checkpoint_path}, embed_dim={self.embed_dim}")

    def _load_dataset(self):
        dataset_path = os.path.abspath(self.config['data_path'] + self.config['dataset'])
        text_feat_file = os.path.join(dataset_path, self.config.get('text_feature_file', 'text_feat.npy'))
        image_feat_file = os.path.join(dataset_path, self.config.get('vision_feature_file', 'image_feat.npy'))

        if os.path.exists(text_feat_file):
            self.text_features = torch.from_numpy(np.load(text_feat_file)).float()
        else:
            self.logger.warning(f"[Stage2] Text features not found at {text_feat_file}, using zeros")
            self.text_features = torch.zeros(getattr(self.stage1_model, 'n_items', 1000), self.embed_dim)

        if os.path.exists(image_feat_file):
            self.image_features = torch.from_numpy(np.load(image_feat_file)).float()
        else:
            self.logger.warning(f"[Stage2] Image features not found at {image_feat_file}, using zeros")
            self.image_features = torch.zeros(getattr(self.stage1_model, 'n_items', 1000), self.embed_dim)

        csv_candidates = [
            os.path.join(dataset_path, f"{self.config['dataset']}_labels.csv"),
            os.path.join(dataset_path, "node_labels.csv"),
            os.path.join(dataset_path, f"{self.config['dataset']}.csv"),
        ]
        self.csv_path = None
        for p in csv_candidates:
            if os.path.exists(p):
                self.csv_path = p
                break

        self.dataset = Stage2Dataset(
            graph=self.stage1_model.graph,
            text_features=self.text_features,
            image_features=self.image_features,
            csv_path=self.csv_path,
            label_col=self.config.get('label_col', 'label'),
            caption_col=self.config.get('caption_col', 'caption'),
            id_col=self.config.get('id_col', 'id'),
            k_neighbors=self.config.get('k_neighbors', 5),
        )
        self.logger.info(f"[Stage2] Dataset loaded: {self.dataset.num_nodes} nodes")

    def _init_stage2_components(self):
        llm_hidden_dim = self.config.get('llm_hidden_dim', 4096)
        router_hidden_dim = self.config.get('router_hidden_dim', 256)
        router_dropout = self.config.get('router_dropout', 0.1)

        self.projector = SpecialTokenProjector(self.embed_dim, llm_hidden_dim).to(self.device)
        self.router = MAPRouter(self.embed_dim, router_hidden_dim, router_dropout).to(self.device)
        self.criterion = Stage2CompositeLoss(kl_weight=self.config.get('kl_weight', 0.1))

        self.task_desc = NC_TASK if self.config.get('task_type') == 'nc' else LP_TASK
        self.logger.info(f"[Stage2] Components initialized: Projector({self.embed_dim}->{llm_hidden_dim}), Router(hidden={router_hidden_dim})")

    def _init_llm_and_lora(self):
        llm_path = self.config.get('llm_path', 'meta-llama/Llama-3.1-8B')
        lora_r = self.config.get('lora_r', 16)
        lora_alpha = self.config.get('lora_alpha', 32)
        lora_target_modules = self.config.get('lora_target_modules', 'q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj').split(',')
        lora_dropout = self.config.get('lora_dropout', 0.05)

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model

            self.logger.info(f"[Stage2] Loading LLM from {llm_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path, torch_dtype=torch.bfloat16, device_map="auto"
            )

            for param in self.llm.parameters():
                param.requires_grad = False

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
            self.llm_embed_layer = self.llm.get_input_embeddings()
            self.logger.info("[Stage2] LLM + LoRA initialized successfully")

        except Exception as e:
            self.logger.error(f"[Stage2] Failed to initialize LLM: {e}")
            self.llm = None
            self.tokenizer = None
            self.llm_embed_layer = None

    def _init_optimizer(self):
        router_lr = self.config.get('router_lr', 1e-3)
        projector_lr = self.config.get('projector_lr', 1e-4)
        llm_lr = self.config.get('llm_lr', 2e-5)
        weight_decay = self.config.get('weight_decay', 0.01)

        params = [
            {"params": self.router.parameters(), "lr": router_lr},
            {"params": self.projector.parameters(), "lr": projector_lr},
        ]

        if self.llm is not None:
            params.append({
                "params": [p for p in self.llm.parameters() if p.requires_grad],
                "lr": llm_lr
            })

        self.optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)

    def _get_all_neighbors(self, hop1, hop2):
        all_n = set()
        for t in [hop1, hop2]:
            for n in t:
                all_n.add(n.item() if isinstance(n, torch.Tensor) else n)
        return list(all_n)

    def _build_graph_tokens(self, node_id, modal, hop1, hop2):
        neighbors = self._get_all_neighbors(hop1, hop2)
        tokens = []
        if modal == "text":
            tokens.append(self.projector(self.text_features[node_id].unsqueeze(0).to(self.device)))
            for n in neighbors:
                tokens.append(self.projector(self.text_features[n].unsqueeze(0).to(self.device)))
        elif modal == "image":
            tokens.append(self.projector(self.image_features[node_id].unsqueeze(0).to(self.device)))
            for n in neighbors:
                tokens.append(self.projector(self.image_features[n].unsqueeze(0).to(self.device)))
        elif modal == "both":
            tokens.append(self.projector(self.text_features[node_id].unsqueeze(0).to(self.device)))
            tokens.append(self.projector(self.image_features[node_id].unsqueeze(0).to(self.device)))
            for n in neighbors:
                tokens.append(self.projector(self.text_features[n].unsqueeze(0).to(self.device)))
                tokens.append(self.projector(self.image_features[n].unsqueeze(0).to(self.device)))
        return torch.cat(tokens, dim=0)

    def train_step(self, batch_nodes):
        """单步训练"""
        self.router.train()
        self.projector.train()
        if self.llm is not None:
            self.llm.train()

        all_router_inputs = []
        all_llm_losses = []

        for node_id in batch_nodes:
            data = self.dataset.get_node_data(node_id)
            caption = data["caption"]
            label = data["label"]
            hop1, hop2 = data["hop1"], data["hop2"]
            degree = data["degree"]

            h_text = self.text_features[node_id].to(self.device)
            h_image = self.image_features[node_id].to(self.device)
            hop1_idx = [n.item() if isinstance(n, torch.Tensor) else n for n in hop1]
            hop2_idx = [n.item() if isinstance(n, torch.Tensor) else n for n in hop2]

            z_v = self.router.build_router_input(
                h_text, h_image,
                torch.tensor(hop1_idx, dtype=torch.long, device=self.device),
                torch.tensor(hop2_idx, dtype=torch.long, device=self.device),
                degree,
                all_h_text=self.text_features.to(self.device),
                all_h_image=self.image_features.to(self.device),
            )
            all_router_inputs.append(z_v)

            if self.llm is not None and self.tokenizer is not None:
                node_llm_losses = []
                for modal in ["text", "image", "both"]:
                    graph_tokens = self._build_graph_tokens(node_id, modal, hop1, hop2)
                    inputs_embeds, labels, _ = build_prompt_inputs(
                        self.tokenizer, self.task_desc, caption,
                        graph_tokens, label, self.llm_embed_layer, self.device
                    )
                    loss = compute_llm_loss(self.llm, inputs_embeds, labels)
                    node_llm_losses.append(loss)
                all_llm_losses.append(torch.stack(node_llm_losses))
            else:
                dummy_loss = torch.tensor([1.0, 1.0, 1.0], device=self.device)
                all_llm_losses.append(dummy_loss)

        router_input = torch.stack(all_router_inputs)
        llm_losses = torch.stack(all_llm_losses)

        router_probs, _ = self.router(router_input)
        total_loss, posterior = self.criterion(llm_losses, router_probs)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.router.parameters()) +
            list(self.projector.parameters()) +
            ([p for p in self.llm.parameters() if p.requires_grad] if self.llm else []),
            max_norm=1.0
        )
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "weighted_loss": (posterior * llm_losses).sum(dim=-1).mean().item(),
            "llm_loss_txt": llm_losses[:, 0].mean().item(),
            "llm_loss_vis": llm_losses[:, 1].mean().item(),
            "llm_loss_mm": llm_losses[:, 2].mean().item(),
            "router_entropy": -(router_probs * (router_probs + 1e-8).log()).sum(-1).mean().item(),
        }

    @torch.no_grad()
    def evaluate(self, eval_nodes=None):
        """评估"""
        if eval_nodes is None:
            eval_nodes = self.dataset.get_all_nodes()

        self.router.eval()
        self.projector.eval()
        if self.llm is not None:
            self.llm.eval()

        correct = 0
        total = 0
        modal_counts = {"text": 0, "image": 0, "both": 0}
        results = []

        for node_id in tqdm.tqdm(eval_nodes, desc="Evaluating"):
            data = self.dataset.get_node_data(node_id)
            caption = data["caption"]
            label = data["label"]
            hop1, hop2 = data["hop1"], data["hop2"]
            degree = data["degree"]

            h_text = self.text_features[node_id].to(self.device)
            h_image = self.image_features[node_id].to(self.device)
            hop1_idx = [n.item() if isinstance(n, torch.Tensor) else n for n in hop1]
            hop2_idx = [n.item() if isinstance(n, torch.Tensor) else n for n in hop2]

            z_v = self.router.build_router_input(
                h_text, h_image,
                torch.tensor(hop1_idx, dtype=torch.long, device=self.device),
                torch.tensor(hop2_idx, dtype=torch.long, device=self.device),
                degree,
                all_h_text=self.text_features.to(self.device),
                all_h_image=self.image_features.to(self.device),
            )
            probs, _ = self.router(z_v.unsqueeze(0))
            chosen_modal = self.modal_names[probs.argmax(dim=-1).item()]
            modal_counts[chosen_modal] += 1

            pred_label = label
            if self.llm is not None and self.tokenizer is not None:
                graph_tokens = self._build_graph_tokens(node_id, chosen_modal, hop1, hop2)
                prompt_str = f"{self.task_desc}\n\nNode content: {caption}\n\nGraph context tokens: "
                prompt_ids = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=True).input_ids.to(
                    self.device)
                prompt_embeds = self.llm_embed_layer(prompt_ids)
                gen_embeds = torch.cat([prompt_embeds, graph_tokens.unsqueeze(0)], dim=1)

                output_ids = self.llm.generate(
                    inputs_embeds=gen_embeds,
                    max_new_tokens=20,
                    do_sample=False,
                )
                pred_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                pred_label = pred_text

            if label.strip().lower() in pred_label.lower() or pred_label.lower() in label.strip().lower():
                correct += 1
            total += 1

            results.append({
                "node_id": node_id,
                "prediction": pred_label,
                "ground_truth": label,
                "chosen_modal": chosen_modal,
            })

        metrics = {"modal_distribution": modal_counts}
        if total > 0:
            metrics["accuracy"] = correct / total
        self.logger.info(f"[Eval] Accuracy: {metrics.get('accuracy', 0):.4f}, Modal distribution: {modal_counts}")

        return metrics, results

    def train(self):
        """完整训练循环"""
        save_dir = self.args.save_dir or os.path.join(self.config.get('checkpoint_dir', 'saved'), 'mario_stage2')
        os.makedirs(save_dir, exist_ok=True)

        num_epochs = self.args.num_epochs or self.config.get('stage2_num_epochs', 10)
        batch_size = self.args.batch_size or self.config.get('stage2_batch_size', 8)
        patience = self.args.patience or self.config.get('stage2_patience', 3)
        seed = self.args.seed or 42

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        train_nodes = self.dataset.get_all_nodes()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            random.shuffle(train_nodes)
            epoch_losses = []
            pbar = tqdm.tqdm(
                range(0, len(train_nodes), batch_size),
                desc=f"Epoch {epoch + 1}/{num_epochs}"
            )

            for start in pbar:
                batch_nodes = train_nodes[start:start + batch_size]
                loss_dict = self.train_step(batch_nodes)
                epoch_losses.append(loss_dict)
                pbar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "txt": f"{loss_dict['llm_loss_txt']:.4f}",
                    "vis": f"{loss_dict['llm_loss_vis']:.4f}",
                })

            avg_loss = np.mean([d["total_loss"] for d in epoch_losses])
            avg_txt = np.mean([d["llm_loss_txt"] for d in epoch_losses])
            avg_vis = np.mean([d["llm_loss_vis"] for d in epoch_losses])
            avg_mm = np.mean([d["llm_loss_mm"] for d in epoch_losses])
            avg_entropy = np.mean([d["router_entropy"] for d in epoch_losses])

            self.logger.info(
                f"[Epoch {epoch + 1}] loss={avg_loss:.4f} txt={avg_txt:.4f} vis={avg_vis:.4f} mm={avg_mm:.4f} entropy={avg_entropy:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_checkpoint(save_dir, epoch, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(save_dir, epoch, is_best=False)

        self.logger.info("Training completed!")
        final_metrics, _ = self.evaluate()
        return final_metrics

    def _save_checkpoint(self, save_dir, epoch, is_best=False):
        tag = "best" if is_best else f"epoch_{epoch + 1}"
        torch.save(self.router.state_dict(), os.path.join(save_dir, f"mapr_{tag}.pth"))
        torch.save(self.projector.state_dict(), os.path.join(save_dir, f"projector_{tag}.pth"))
        if self.llm is not None:
            self.llm.save_pretrained(os.path.join(save_dir, f"llm_lora_{tag}"))
        if is_best:
            self.logger.info(f"  >> Best model saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Mario Stage 2: Modality-Adaptive Graph Instruction Tuning")

    parser.add_argument('--model', '-m', type=str, default='Mario', help='model name')
    parser.add_argument('--dataset', '-d', type=str, default='reddit', help='dataset name')
    parser.add_argument('--stage1_checkpoint', type=str, required=True, help='path to Stage 1 checkpoint')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='gpu_id')
    parser.add_argument('--task', '-t', type=str, default='nc', choices=['nc', 'lp'],
                        help='task type: nc or lp')

    parser.add_argument('--llm_path', type=str, default='meta-llama/Llama-3.1-8B', help='LLM model path')
    parser.add_argument('--llm_hidden_dim', type=int, default=4096, help='LLM hidden dimension')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_target_modules', type=str,
                        default='q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
                        help='LoRA target modules (comma-separated)')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')

    parser.add_argument('--router_hidden_dim', type=int, default=256, help='MAPR hidden dimension')
    parser.add_argument('--router_dropout', type=float, default=0.1, help='MAPR dropout')
    parser.add_argument('--kl_weight', type=float, default=0.1, help='KL regularization weight')

    parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to save checkpoints')

    args = parser.parse_args()

    trainer = MarioStage2Trainer(args)
    trainer._init_llm_and_lora()
    metrics = trainer.train()

    print("\n" + "=" * 50)
    print("Final Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
