# coding: utf-8

"""
Mario Stage 2: Modality-Adaptive Graph Instruction Tuning
模态自适应图指令微调组件

核心组件:
- MAPRouter: 模态自适应路由器，为每个节点选择最优模态模板
- Stage2CompositeLoss: 复合损失函数 (性能加权 + KL正则化)
- PromptTemplate: 基础提示模板构建器
- MarioPromptTemplate: 完整提示模板（含任务描述和原始文本）
- NC_TASK / LP_TASK: 节点分类/链接预测任务描述
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl


NC_TASK = '''
Node classification is a fundamental task in graph machine learning. Given a graph G = (V, E), where V is the set of nodes and E is the set of edges, each node v in V may be associated with a feature vector x_v. In node classification, a subset of nodes V_train subset of V is labeled with ground-truth categories (e.g., user interests, document topics, or protein functions). The goal is to predict the missing labels for the remaining nodes V_unlabeled by leveraging:

- The node features (if available) – intrinsic attributes of each node.
- The graph structure – connections between nodes encode relationships, dependencies, or interactions that often imply label similarity or influence.

This task is semi-supervised by nature, as only a small fraction of nodes typically have labels. Models learn to propagate label information through edges, using inductive biases like homophily (connected nodes tend to share the same label) or structural equivalence. Common approaches include graph neural networks (e.g., GCN, GraphSAGE, GAT), label propagation algorithms, and graph-based regularization.

Applications range from social network analysis (predicting user attributes), citation network classification (paper topics), fraud detection (identifying malicious accounts), to knowledge graph reasoning and biological network analysis.
'''

LP_TASK = '''
Link prediction aims to infer missing or future connections between nodes in a graph. Given a graph G = (V, E), where V is the set of nodes and E is the set of observed edges, the task is to predict which unobserved node pairs (u, v) not in E are likely to form an edge (or will form one in a temporal setting).

The problem is typically framed as a binary classification or ranking task:
- Positive samples: existing edges (or edges that appear in a future time window).
- Negative samples: randomly sampled non-edges (or edges that never appear).

Key information sources:
- Topological structure: Common neighbors, Jaccard coefficient, Adamic–Adar, preferential attachment, and Katz index capture local and global graph patterns.
- Node features: If available, similarity or interaction between feature vectors can be used.
- Latent representations: Graph neural networks (e.g., GAE, VGAE, SEAL) learn node embeddings that preserve both structural and feature proximity, then predict links via a decoder (e.g., inner product or MLP on node pairs).
'''


class MAPRouter(nn.Module):
    """
    Modality-Adaptive Prompt Router (MAPR)
    输入: z_v = [h_text_v; h_image_v; phi^(1)(v); phi^(2)(v); log(d_v)]  in R^{4d+1}
    输出: s_v in R^3 -> softmax -> p_v = [p_txt, p_vis, p_mm]
    """

    def __init__(self, embed_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        input_dim = 4 * embed_dim + 1

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def build_router_input(self, h_text, h_image, neighbors_1_indices, neighbors_2_indices,
                           degrees, all_h_text=None, all_h_image=None):
        device = h_text.device

        if len(neighbors_1_indices) > 0 and all_h_text is not None:
            n1_text = all_h_text[neighbors_1_indices]
            n1_image = all_h_image[neighbors_1_indices]
            phi1 = (n1_text + n1_image).mean(dim=0)
        else:
            phi1 = torch.zeros_like(h_text)

        if len(neighbors_2_indices) > 0 and all_h_text is not None:
            n2_text = all_h_text[neighbors_2_indices]
            n2_image = all_h_image[neighbors_2_indices]
            phi2 = (n2_text + n2_image).mean(dim=0)
        else:
            phi2 = torch.zeros_like(h_text)

        log_degree = torch.tensor(
            [torch.log(torch.tensor(float(degrees) + 1.0))],
            device=device, dtype=h_text.dtype
        )

        z_v = torch.cat([h_text, h_image, phi1, phi2, log_degree], dim=0)
        return z_v

    def forward(self, z):
        logits = self.router(z)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class Stage2CompositeLoss(nn.Module):
    """
    Stage 2 复合损失函数
    L_S2 = (1/|B|) * sum_v [ sum_k q_v^(k) * ell_v^(k) + lambda * KL(q_v || p_v) ]

    其中:
    - ell_v^(k): LLM 对模板 k 的负对数似然损失
    - q_v = softmax(-[ell_txt, ell_vis, ell_mm]): 性能后验
    - p_v = softmax(s_v): 路由器输出概率
    - lambda: KL 正则化系数
    """

    def __init__(self, kl_weight=0.1):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, llm_losses, router_probs):
        posterior = F.softmax(-llm_losses, dim=-1)
        weighted_loss = (posterior * llm_losses).sum(dim=-1).mean()
        kl_div = F.kl_div(
            router_probs.log(),
            posterior,
            reduction='batchmean'
        )
        total_loss = weighted_loss + self.kl_weight * kl_div
        return total_loss, posterior


class SpecialTokenProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpecialTokenProjector, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class PromptTemplate(object):
    """
    基于图结构的 Prompt Template 构建器
    使用 Top-K 余弦相似度选择邻居节点
    """

    def __init__(self, text_nodefeatures, image_nodefeatures, center, k_neighbors=5,
                 hop1_neighbors=None, hop2_neighbors=None):
        self.text_nodefeatures = text_nodefeatures
        self.image_nodefeatures = image_nodefeatures
        self.center = center
        self.k = k_neighbors
        self.hop1_neighbors = hop1_neighbors if hop1_neighbors is not None else []
        self.hop2_neighbors = hop2_neighbors if hop2_neighbors is not None else []

    def generate_template(self, modal="text", neighbor_hop_number=None):
        if neighbor_hop_number is None:
            neighbors = list(set(
                [n.item() if isinstance(n, torch.Tensor) else n for n in self.hop1_neighbors] +
                [n.item() if isinstance(n, torch.Tensor) else n for n in self.hop2_neighbors]
            ))
        elif neighbor_hop_number == 1:
            neighbors = [n.item() if isinstance(n, torch.Tensor) else n for n in self.hop1_neighbors]
        elif neighbor_hop_number == 2:
            neighbors = [n.item() if isinstance(n, torch.Tensor) else n for n in self.hop2_neighbors]
        else:
            raise ValueError("neighbor_hop_number must be None, 1 or 2")

        prompt_parts = []

        if modal == "text":
            prompt_parts.append(self.text_nodefeatures[self.center])
            for n in neighbors:
                prompt_parts.append(self.text_nodefeatures[n])

        elif modal == "image":
            prompt_parts.append(self.image_nodefeatures[self.center])
            for n in neighbors:
                prompt_parts.append(self.image_nodefeatures[n])

        elif modal == "both":
            prompt_parts.append(self.text_nodefeatures[self.center])
            prompt_parts.append(self.image_nodefeatures[self.center])
            for n in neighbors:
                prompt_parts.append(self.text_nodefeatures[n])
                prompt_parts.append(self.image_nodefeatures[n])

        else:
            raise ValueError("modal must be 'text', 'image' or 'both'")

        return torch.stack(prompt_parts, dim=0)


class MarioPromptTemplate(PromptTemplate):
    """
    Mario 完整 Prompt Template，包含任务描述和原始文本
    """

    def __init__(self, text_nodefeatures, image_nodefeatures, center,
                 task_description=None, raw_caption=None, k_neighbors=5,
                 hop1_neighbors=None, hop2_neighbors=None):
        super(MarioPromptTemplate, self).__init__(
            text_nodefeatures, image_nodefeatures, center, k_neighbors,
            hop1_neighbors, hop2_neighbors
        )

        if task_description is None:
            self.task_description = NC_TASK
        else:
            if task_description.lower() in ['nc', 'node_classification']:
                self.task_description = NC_TASK
            elif task_description.lower() in ['lp', 'link_prediction']:
                self.task_description = LP_TASK
            else:
                self.task_description = task_description

        self.raw_caption = raw_caption or ""

    def build_full_prompt(self, modal="text", neighbor_hop_number=None):
        special_tokens = self.generate_template(modal=modal, neighbor_hop_number=neighbor_hop_number)
        return {
            "task_description": self.task_description,
            "raw_caption": self.raw_caption,
            "special_tokens": special_tokens,
        }


class Stage2Dataset:
    """
    Stage 2 数据集管理器
    管理: 图结构、Stage 1 特征、节点 caption/label、Top-K 邻居
    """

    def __init__(self, graph, text_features, image_features, csv_path=None,
                 label_col="label", caption_col="caption", id_col="id",
                 k_neighbors=5):
        self.graph = graph
        self.text_features = text_features
        self.image_features = image_features
        self.num_nodes = text_features.size(0)
        self.k = k_neighbors
        self.label_col = label_col
        self.caption_col = caption_col
        self.id_col = id_col

        self.csv_data = None
        if csv_path and os.path.exists(csv_path):
            self.csv_data = pd.read_csv(csv_path)

        print("[Stage2] Precomputing Top-K neighbors...")
        self.hop1_neighbors = {}
        self.hop2_neighbors = {}
        self.degrees = {}

        for node_id in range(min(self.num_nodes, 10000)):
            self._compute_neighbors(node_id)

        if self.num_nodes > 10000:
            print(f"[Stage2] Warning: Only precomputed neighbors for first 10000 of {self.num_nodes} nodes")

    def _compute_neighbors(self, node_id):
        g = self.graph

        succ = g.successors(node_id) if hasattr(g, 'successors') else g.successors(node_id)
        if len(succ) > self.k:
            sim = self._neighbor_similarity(node_id, succ)
            _, idx = torch.topk(sim, self.k)
            succ = succ[idx]
        self.hop1_neighbors[node_id] = succ
        self.degrees[node_id] = len(g.successors(node_id)) if hasattr(g, 'successors') else len(
            g.successors(node_id))

        hop2_set = set()
        for n in self.hop1_neighbors[node_id]:
            n_item = n.item() if isinstance(n, torch.Tensor) else n
            try:
                for h in g.successors(n_item):
                    h_item = h.item() if isinstance(h, torch.Tensor) else h
                    if h_item != node_id:
                        hop2_set.add(h_item)
            except:
                pass
        hop2 = torch.tensor(list(hop2_set), dtype=torch.long) if hop2_set else torch.tensor([], dtype=torch.long)
        if len(hop2) > self.k:
            sim = self._neighbor_similarity(node_id, hop2)
            _, idx = torch.topk(sim, self.k)
            hop2 = hop2[idx]
        self.hop2_neighbors[node_id] = hop2

    def _neighbor_similarity(self, center, candidates):
        center_feat = torch.cat([
            self.text_features[center], self.image_features[center]
        ], dim=-1).unsqueeze(0)
        cand_feats = torch.cat([
            self.text_features[candidates], self.image_features[candidates]
        ], dim=-1)
        return F.cosine_similarity(center_feat, cand_feats, dim=1)

    def get_node_data(self, node_id):
        data = {
            "node_id": node_id,
            "hop1": self.hop1_neighbors.get(node_id, torch.tensor([], dtype=torch.long)),
            "hop2": self.hop2_neighbors.get(node_id, torch.tensor([], dtype=torch.long)),
            "degree": self.degrees.get(node_id, 0),
        }

        if self.csv_data is not None:
            row = self.csv_data[self.csv_data[self.id_col] == node_id]
            if len(row) > 0:
                row = row.iloc[0]
                data["caption"] = str(row.get(self.caption_col, ""))
                data["label"] = str(row.get(self.label_col, ""))
            else:
                data["caption"] = ""
                data["label"] = ""
        else:
            data["caption"] = ""
            data["label"] = ""

        return data

    def get_all_nodes(self):
        return list(range(self.num_nodes))


def build_prompt_inputs(tokenizer, task_description, raw_caption, special_token_embeds,
                        answer_text, llm_embed_layer, device):
    """
    构造 LLM 输入: [task_tokens; caption_tokens; special_graph_tokens] + [answer_tokens]

    Returns:
        inputs_embeds: (1, total_seq_len, hidden_dim)
        labels: (1, total_seq_len) prompt 部分为 -100
        prompt_len: int
    """
    prompt_str = f"{task_description}\n\nNode content: {raw_caption}\n\nGraph context tokens: "
    answer_str = f" {answer_text}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    answer_ids = tokenizer(answer_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prompt_embeds = llm_embed_layer(prompt_ids)
    answer_embeds = llm_embed_layer(answer_ids)

    graph_tokens = special_token_embeds.unsqueeze(0)
    inputs_embeds = torch.cat([prompt_embeds, graph_tokens, answer_embeds], dim=1)

    prompt_len = prompt_ids.size(1) + special_token_embeds.size(0)

    labels = torch.full((1, inputs_embeds.size(1)), -100, dtype=torch.long, device=device)
    labels[0, prompt_len:] = answer_ids[0]

    return inputs_embeds, labels, prompt_len


def compute_llm_loss(llm_model, inputs_embeds, labels):
    """计算 LLM 的 causal LM loss"""
    outputs = llm_model(inputs_embeds=inputs_embeds, labels=labels)
    return outputs.loss
