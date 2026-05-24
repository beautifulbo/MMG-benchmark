# coding: utf-8

"""
Mario: Multimodal Graph Learning with Distance-aware Attention
Stage 1 - Feature Extractor for Missing Modality Scenarios

核心组件:
- MultiHeadAttention: 带图距离感知位置编码的多头注意力
- RepeatedMultiHeadAttention: 堆叠多层注意力
- FeatureExtractor: 单模态特征提取通道
- MarioFeatureExtractor: 双通道（文本+图像）特征提取器
- Mario: 主模型类，继承 GeneralRecommender，兼容 MMG-benchmark
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .base.abstract_recommender import GeneralRecommender


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, graph, buckets_num=6, global_dist_matrix=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.graph = graph
        self.buckets_num = buckets_num
        self.dist_matrix = global_dist_matrix.cpu()

        self.buckets = nn.ParameterList([
            nn.Parameter(torch.randn(buckets_num)) for _ in range(num_heads)
        ])

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_linear_1 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_linear_2 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(embed_dim)

    @torch.compiler.disable
    def _compute_position_embedding_batch(self, idx, device):
        idx_list = idx.cpu().tolist()
        dist_matrix = self.dist_matrix[idx_list][:, idx_list]
        dist_matrix = dist_matrix.clamp(min=0, max=self.buckets_num - 1).long()
        dist_matrix = dist_matrix.to(device, non_blocking=True)
        buckets_stacked = torch.stack([b for b in self.buckets], dim=0)
        dist_expanded = dist_matrix.unsqueeze(0).expand(self.num_heads, -1, -1)
        pos_emb = buckets_stacked.gather(1, dist_expanded.reshape(self.num_heads, -1)).reshape(
            self.num_heads, -1, dist_matrix.size(1))
        return pos_emb.to(device)

    def forward(self, x, idx):
        batch_size, seq_len, _ = x.size()
        pos_emb = self._compute_position_embedding_batch(idx, x.device)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + pos_emb.unsqueeze(0)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear_1(attn_output)
        output = self.gelu(output)
        output = self.out_linear_2(output)
        output = self.ln(output + x)
        h = output[:, 0:1, :]
        x1 = x[:, 1:, :]
        output = torch.cat([h, x1], dim=1)
        return output


class RepeatedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, graph, buckets_num=6, layers=6, global_dist_matrix=None):
        super(RepeatedMultiHeadAttention, self).__init__()
        self.layers = layers
        self.attention = nn.ModuleList(
            [MultiHeadAttention(embed_dim, num_heads, graph, buckets_num, global_dist_matrix) for _ in
             range(layers)])

    def forward(self, x, idx):
        for i in range(self.layers):
            x = self.attention[i](x, idx)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads, graph, buckets_num=6, layers=6, global_dist_matrix=None):
        super(FeatureExtractor, self).__init__()
        self.repeated_attention = RepeatedMultiHeadAttention(embed_dim, num_heads, graph, buckets_num, layers,
                                                             global_dist_matrix)

    def forward(self, x, idx):
        return self.repeated_attention(x, idx)


class MarioFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads, graph, buckets_num=6, layers=6, dist_matrix_path=None,
                 global_dist_matrix=None):
        super(MarioFeatureExtractor, self).__init__()
        if global_dist_matrix is not None:
            self._dist_matrix = global_dist_matrix
        elif dist_matrix_path is not None and os.path.exists(dist_matrix_path):
            print(f"[Mario] Loading global distance matrix from {dist_matrix_path} ...")
            self._dist_matrix = torch.load(dist_matrix_path, mmap=True, weights_only=True)
            print(f"[Mario] Loaded! Shape: {self._dist_matrix.shape}")
        else:
            raise ValueError("Must provide either global_dist_matrix or valid dist_matrix_path")

        self.text_feature_extractor = FeatureExtractor(embed_dim, num_heads, graph, buckets_num, layers,
                                                       self._dist_matrix)
        self.image_feature_extractor = FeatureExtractor(embed_dim, num_heads, graph, buckets_num, layers,
                                                        self._dist_matrix)

    @property
    def dist_matrix(self):
        return self._dist_matrix

    def forward(self, text, image, idx):
        text_features = self.text_feature_extractor(text, idx)
        image_features = self.image_feature_extractor(image, idx)
        return text_features, image_features


class SpecialTokenProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpecialTokenProjector, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_text, h_image):
        h_text = F.normalize(h_text, dim=1)
        h_image = F.normalize(h_image, dim=1)
        sim = torch.matmul(h_text, h_image.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        loss_t2i = F.cross_entropy(sim, labels)
        loss_i2t = F.cross_entropy(sim.T, labels)
        return (loss_t2i + loss_i2t) / 2.0


class Mario(GeneralRecommender):
    """
    Mario: Multimodal Graph Learning with Distance-aware Attention
    用于缺失模态场景的多模态图学习模型

    Stage 1: 使用图距离感知注意力机制提取多模态节点特征
             通过 InfoNCE loss 进行跨模态对比学习

    Stage 2 (可选): MAPR 路由器 + LLM LoRA 微调 (见 mario_stage2.py)
    """

    def __init__(self, config, dataset):
        super(Mario, self).__init__(config, dataset)

        self.embedding_dim = config.get('embedding_size', 3584)
        self.num_heads = config.get('num_heads', 8)
        self.buckets_num = config.get('buckets_num', 6)
        self.n_layers = config.get('n_layers', 6)
        self.temperature = config.get('temperature', 0.07)
        self.k_neighbors = config.get('k_neighbors', 5)
        self.task_type = config.get('task_type', 'lp')

        self._build_graph(config, dataset)
        self._load_or_compute_distance_matrix(config)
        self._init_feature_extractor()
        self._handle_missing_modality(config)
        self._init_classifier(config)

        self.info_nce_loss = InfoNCELoss(temperature=self.temperature)

    def _build_graph(self, config, dataset):
        graph_path = config.get('graph_path')
        if graph_path and os.path.exists(graph_path):
            print(f"[Mario] Loading graph from {graph_path}")
            self.graph = dgl.load_graphs(graph_path)[0][0]
        else:
            print("[Mario] Building graph from interaction matrix...")
            inter_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
            src_nodes = np.repeat(np.arange(self.n_users), inter_matrix.nnz // self.n_users if inter_matrix.nnz > 0 else 0)
            dst_nodes = []
            if inter_matrix.nnz > 0:
                rows, cols = inter_matrix.row, inter_matrix.col
                for r, c in zip(rows, cols):
                    dst_nodes.append(c)
                src_nodes = np.array(rows[:len(dst_nodes)])
                dst_nodes = np.array(dst_nodes)
            if len(src_nodes) > 0:
                self.graph = dgl.graph((src_nodes, dst_nodes), num_nodes=self.n_users + self.n_items)
            else:
                self.graph = dgl.graph(([], []), num_nodes=self.n_users + self.n_items)
        print(f"[Mario] Graph built: {self.graph.num_nodes()} nodes, {self.graph.num_edges()} edges")

    def _load_or_compute_distance_matrix(self, config):
        dist_matrix_path = config.get('dist_matrix_path')
        cache_dir = config.get('checkpoint_dir', 'saved')

        if dist_matrix_path and os.path.exists(dist_matrix_path):
            print(f"[Mario] Loading pre-computed distance matrix from {dist_matrix_path}")
            self.global_dist_matrix = torch.load(dist_matrix_path, mmap=True, weights_only=True)
        else:
            print("[Mario] Computing shortest path distance matrix...")
            self.global_dist_matrix = self._compute_shortest_path_matrix(self.graph)
            if self.global_dist_matrix is not None:
                cache_path = os.path.join(cache_dir, 'mario_dist_matrix.pt')
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(self.global_dist_matrix, cache_path)
                print(f"[Mario] Distance matrix cached to {cache_path}")

    def _compute_shortest_path_matrix(self, graph):
        try:
            import networkx as nx
            nx_graph = dgl.to_networkx(graph).to_undirected()
            num_nodes = graph.num_nodes()
            if num_nodes > 5000:
                print(f"[Mario] Warning: Graph has {num_nodes} nodes, using approximate distances")
                dist_matrix = torch.full((num_nodes, num_nodes), self.buckets_num - 1, dtype=torch.long)
                for node_id in range(min(num_nodes, 1000)):
                    try:
                        lengths = dict(nx.single_source_shortest_path_length(nx_graph, node_id, cutoff=self.buckets_num - 1))
                        for target, length in lengths.items():
                            dist_matrix[node_id, target] = min(length, self.buckets_num - 1)
                    except:
                        pass
                return dist_matrix
            else:
                dist_dict = dict(nx.all_pairs_shortest_path_length(nx_graph))
                dist_matrix = torch.full((num_nodes, num_nodes), self.buckets_num - 1, dtype=torch.long)
                for src in range(num_nodes):
                    if src in dist_dict:
                        for tgt, length in dist_dict[src].items():
                            dist_matrix[src, tgt] = min(int(length), self.buckets_num - 1)
                return dist_matrix
        except ImportError:
            print("[Mario] NetworkX not available, using identity-based distance matrix")
            num_nodes = graph.num_nodes()
            return torch.zeros(num_nodes, num_nodes, dtype=torch.long)
        except Exception as e:
            print(f"[Mario] Error computing distance matrix: {e}, using fallback")
            num_nodes = graph.num_nodes()
            return torch.full((num_nodes, num_nodes), 3, dtype=torch.long)

    def _init_feature_extractor(self):
        actual_embed_dim = self._get_actual_embed_dim()
        self.feature_extractor = MarioFeatureExtractor(
            embed_dim=actual_embed_dim,
            num_heads=min(self.num_heads, actual_embed_dim // 64) if actual_embed_dim >= 128 else max(1,
                                                                                                  actual_embed_dim // 64),
            graph=self.graph,
            buckets_num=self.buckets_num,
            layers=self.n_layers,
            global_dist_matrix=self.global_dist_matrix
        ).to(self.device)
        print(f"[Mario] MarioFeatureExtractor initialized: dim={actual_embed_dim}, heads={self.num_heads}, layers={self.n_layers}")

    def _get_actual_embed_dim(self):
        feat_dim = None
        if self.v_feat is not None:
            feat_dim = self.v_feat.shape[1]
        if self.t_feat is not None:
            if feat_dim is None:
                feat_dim = self.t_feat.shape[1]
            elif feat_dim != self.t_feat.shape[1]:
                print(
                    f"[Mario] Warning: v_feat dim ({self.v_feat.shape[1]}) != t_feat dim ({self.t_feat.shape[1]}), using v_feat dim")
        if feat_dim is None:
            feat_dim = self.embedding_dim
            print(f"[Mario] No features loaded, using configured embedding_size={feat_dim}")
        return feat_dim

    def _handle_missing_modality(self, config):
        self.missing_modal = config.get('missing_modal', False)
        if self.missing_modal:
            missing_ratio = config.get('missing_ratio', 0.3)
            missing_modality_type = config.get('missing_modality_type', 'all')
            missing_strategy = config.get('missing_strategy', 'mean')
            self._apply_missing_modality(missing_ratio, missing_modality_type, missing_strategy)

    def _apply_missing_modality(self, ratio, modality_type, strategy):
        n_items = self.n_items
        n_missing = int(n_items * ratio)
        np.random.seed(42)
        if modality_type in ['all', 'text']:
            missing_t_idx = np.random.choice(n_items, n_missing, replace=False)
            if strategy == 'mean':
                non_missing_t = np.setdiff1d(np.arange(n_items), missing_t_idx)
                if len(non_missing_t) > 0 and self.t_feat is not None:
                    mean_t = self.t_feat[non_missing_t].mean(dim=0)
                    self.t_feat[missing_t_idx] = mean_t
            elif strategy == 'zero' and self.t_feat is not None:
                self.t_feat[missing_t_idx] = 0.0
            print(f"[Mario] Applied missing TEXT modality: {len(missing_t_idx)} items ({strategy} fill)")
        if modality_type in ['all', 'image']:
            missing_v_idx = np.random.choice(n_items, n_missing, replace=False)
            if strategy == 'mean':
                non_missing_v = np.setdiff1d(np.arange(n_items), missing_v_idx)
                if len(non_missing_v) > 0 and self.v_feat is not None:
                    mean_v = self.v_feat[non_missing_v].mean(dim=0)
                    self.v_feat[missing_v_idx] = mean_v
            elif strategy == 'zero' and self.v_feat is not None:
                self.v_feat[missing_v_idx] = 0.0
            print(f"[Mario] Applied missing IMAGE modality: {len(missing_v_idx)} items ({strategy} fill)")

    def _init_classifier(self, config):
        self.use_stage2 = config.get('use_stage2', False)
        self.num_classes = config.get('num_classes', None)
        actual_dim = self._get_actual_embed_dim()
        self.nc_classifier = nn.Sequential(
            nn.Linear(actual_dim, actual_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(actual_dim // 2, self.num_classes if self.num_classes else 10)
        ).to(self.device) if self.task_type == 'nc' else None
        if self.nc_classifier is not None:
            print(f"[Mario] Node classification classifier initialized: {actual_dim} -> {self.num_classes or 10}")

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction
        all_item_ids = torch.unique(torch.cat([pos_items, neg_items]))
        batch_size = users.size(0)

        text_feats, image_feats = self._extract_features(all_item_ids)
        text_feats = text_feats.squeeze(0)
        image_feats = image_feats.squeeze(0)
        loss = self.info_nce_loss(text_feats, image_feats)
        return loss

    def full_sort_predict(self, interaction):
        users, _ = interaction
        all_item_ids = torch.arange(self.n_items).to(self.device)
        text_feats, image_feats = self._extract_features(all_item_ids)
        text_feats = text_feats.squeeze(0)
        image_feats = image_feats.squeeze(0)
        combined_feats = (F.normalize(text_feats, dim=1) + F.normalize(image_feats, dim=1)) / 2.0
        user_feats = self._get_user_features(users)
        user_feats = F.normalize(user_feats, dim=1)
        scores = torch.matmul(user_feats, combined_feats.T)
        return scores

    def _extract_features(self, item_ids):
        item_ids_cpu = item_ids.cpu()
        batch_size = len(item_ids)
        actual_dim = self._get_actual_embed_dim()

        if self.t_feat is not None:
            text_input = self.t_feat[item_ids_cpu].unsqueeze(0).to(self.device)
        else:
            text_input = torch.zeros(1, batch_size, actual_dim, device=self.device)
        if self.v_feat is not None:
            image_input = self.v_feat[item_ids_cpu].unsqueeze(0).to(self.device)
        else:
            image_input = torch.zeros(1, batch_size, actual_dim, device=self.device)

        text_input = text_input.view(1, batch_size, -1)
        image_input = image_input.view(1, batch_size, -1)

        text_feats, image_feats = self.feature_extractor(text_input, image_input, item_ids.to(self.device))
        center_text = text_feats[:, 0, :]
        center_image = image_feats[:, 0, :]
        return center_text, center_image

    def _get_user_features(self, user_ids):
        actual_dim = self._get_actual_embed_dim()
        user_interacted_items = []
        for uid in user_ids.cpu().tolist():
            items = torch.where((torch.arange(self.n_items) >= 0))[0][:self.k_neighbors]
            user_interacted_items.append(items)
        batch_item_ids = torch.stack(user_interacted_items).to(self.device)
        text_feats, image_feats = self._extract_features(batch_item_ids.view(-1))
        text_feats = text_feats.view(len(user_ids), self.k_neighbors, -1)
        image_feats = image_feats.view(len(user_ids), self.k_neighbors, -1)
        user_feats = (text_feats.mean(dim=1) + image_feats.mean(dim=1)) / 2.0
        return user_feats

    def get_node_embeddings(self, node_ids=None):
        if node_ids is None:
            node_ids = torch.arange(self.n_items).to(self.device)
        text_feats, image_feats = self._extract_features(node_ids)
        text_feats = text_feats.squeeze(0)
        image_feats = image_feats.squeeze(0)
        return (text_feats + image_feats) / 2.0

    def predict_nc(self, node_ids):
        if self.nc_classifier is None:
            raise ValueError("Node classifier not initialized. Set task_type='nc' in config.")
        embeddings = self.get_node_embeddings(node_ids)
        logits = self.nc_classifier(embeddings)
        return logits

    def init_stage2(self, config):
        from .mario_stage2 import MAPRouter, SpecialTokenProjector
        self.use_stage2 = True
        actual_dim = self._get_actual_embed_dim()
        llm_hidden_dim = config.get('llm_hidden_dim', 4096)
        router_hidden_dim = config.get('router_hidden_dim', 256)
        router_dropout = config.get('router_dropout', 0.1)
        self.stage2_projector = SpecialTokenProjector(actual_dim, llm_hidden_dim).to(self.device)
        self.stage2_router = MAPRouter(actual_dim, router_hidden_dim, router_dropout).to(self.device)
        print(f"[Mario] Stage 2 initialized: Projector({actual_dim}->{llm_hidden_dim}), Router(hidden={router_hidden_dim})")

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def __str__(self):
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
