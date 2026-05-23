# coding: utf-8

"""
CRL-MMNAR: Causal Representation Learning for Multimodal data with 
Missing Not At Random patterns, adapted for recommendation.

Core components adapted from the clinical prediction version:
- ModalityEncoder: Encodes image and text features
- CrossModalAttention with z-gating: Fuses modalities with missing pattern awareness
- GraphModalityFusion: Item-modality bipartite graph for population-level modeling
- RepresentationBalancingModule: Handles missing modality by prediction and calibration

This model is adapted from multimodal_missingness.py to work in MMG-benchmark's
recommendation framework, following DGMRec as a reference implementation.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.abstract_recommender import GeneralRecommender
from .base.loss import MSELoss
from data.utils.graph_utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood


class ModalityEncoder(nn.Module):
    """Encoder for a specific modality."""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal fusion with explicit z-gating mechanism.
    Gates for each modality are computed from z = MLP(δ), not from (feature + pattern).
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-/Cross-attention blocks
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Norms + FFN
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Modality-specific gates: W_m z + b -> sigmoid
        self.modality_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(2)  # 2 modalities: image and text
        ])
        
        # Missing-pattern embeddings for cross-attention
        self.pattern_embeddings = nn.Embedding(4, hidden_dim)  # 4 patterns for 2 modalities
    
    def forward(self, modality_features, missing_flags, missing_patterns, missing_repr):
        """
        Args:
            modality_features: list of 2 tensors [B, D] (image, text)
            missing_flags: [B, 2] bool/int {0,1}
            missing_patterns: [B] (categorical index 0..3)
            missing_repr: [B, D] (z = MLP(δ))
        """
        B = modality_features[0].size(0)
        
        # z-gate: gate_m = sigmoid(W_m z + b), then ẽ_m = δ_m * gate_m * e_m
        gated_features = []
        for i, (feat, gate_m) in enumerate(zip(modality_features, self.modality_gates)):
            gate_w = gate_m(missing_repr)  # [B, D]
            gated = feat * gate_w * missing_flags[:, i:i+1].float()
            gated_features.append(gated)  # 2 × [B, D]
        
        # Stack to [B, 2, D]
        stacked = torch.stack(gated_features, dim=1)
        
        # key_padding_mask: True = pad/ignore; mask out missing modalities
        key_pad_mask = ~(missing_flags.bool())  # [B, 2]
        
        # Self-attention over available modalities
        self_attended, self_attn_w = self.self_attention(
            stacked, stacked, stacked, key_padding_mask=key_pad_mask
        )
        self_attended = self.norm1(self_attended + stacked)
        
        # Cross-attention with pattern embedding
        pat = self.pattern_embeddings(missing_patterns)  # [B, D]
        pat = pat.unsqueeze(1).expand(-1, 2, -1)  # [B, 2, D]
        cross_attended, cross_attn_w = self.cross_attention(
            self_attended, pat, pat, key_padding_mask=key_pad_mask
        )
        cross_attended = self.norm2(cross_attended + self_attended)
        
        # FFN + residual
        out = self.ffn(cross_attended)
        out = self.norm3(out + cross_attended)  # [B, 2, D]
        
        # Availability-weighted aggregation to item-level h_i
        weights = missing_flags.float().unsqueeze(-1)  # [B, 2, 1]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        aggregated = (out * weights).sum(dim=1)  # [B, D]
        
        return aggregated, self_attn_w, cross_attn_w


class GraphModalityFusion(nn.Module):
    """Graph-based fusion inspired by MUSE for item-level MMNAR modeling."""
    
    def __init__(self, hidden_dim, graph_hidden_dim=32, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.num_layers = num_layers
        
        # Item and modality node embeddings
        self.item_proj = nn.Linear(hidden_dim, graph_hidden_dim)
        self.modality_embeddings = nn.Parameter(torch.randn(2, graph_hidden_dim))
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(graph_hidden_dim, graph_hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(graph_hidden_dim, hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, item_features, missing_flags, batch_indices=None):
        batch_size = item_features.size(0)
        device = item_features.device
        
        # Project item features to graph space
        item_graph_features = self.item_proj(item_features)
        
        # Create bipartite graph adjacency
        adjacency = missing_flags.float()
        
        # Expand modality embeddings for the batch
        modality_features = self.modality_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Apply graph attention layers
        current_item_features = item_graph_features
        current_modality_features = modality_features
        
        for layer in self.graph_layers:
            updated_item_features = layer(
                current_item_features,
                current_modality_features,
                adjacency
            )
            current_item_features = updated_item_features
        
        # Project back to original space
        graph_enhanced_features = self.output_proj(current_item_features)
        
        # Residual connection and normalization
        output_features = self.norm(graph_enhanced_features + item_features)
        
        return output_features


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for item-modality bipartite graph."""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(in_dim, num_heads=2, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
    
    def forward(self, item_features, modality_features, adjacency):
        batch_size = item_features.size(0)
        
        # Prepare for attention: item as query, modalities as key/value
        queries = item_features.unsqueeze(1)
        keys = modality_features
        values = modality_features
        
        # Create attention mask from adjacency
        attn_mask = (adjacency == 0)
        
        # Apply attention
        attended_output, attn_weights = self.attention(
            queries, keys, values,
            key_padding_mask=attn_mask
        )
        
        # Remove sequence dimension
        attended_output = attended_output.squeeze(1)
        
        # Residual connection and normalization
        attended_output = self.norm1(attended_output + item_features)
        
        # Feed-forward network
        ffn_output = self.ffn(attended_output)
        
        # Final normalization
        output = self.norm2(ffn_output)
        
        return output


class RepresentationBalancingModule(nn.Module):
    """Representation balancing module for MMNAR patterns (2-modalities version)."""
    
    def __init__(self, hidden_dim, num_modalities=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Modality predictors
        self.modality_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * (num_modalities - 1), hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1 * 0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_modalities)
        ])
        
        # Feature calibration network
        self.calibration = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1 * 0.5),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Contrastive projection head
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1 * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
    
    def forward(self, features, modality_features, missing_flags):
        batch_size = features.size(0)
        
        # Predict each modality from others
        predicted_modalities = []
        
        for i in range(self.num_modalities):
            other_features = []
            for j in range(self.num_modalities):
                if j != i:
                    masked_feature = modality_features[j] * missing_flags[:, j].unsqueeze(1)
                    other_features.append(masked_feature)
            
            others_concat = torch.cat(other_features, dim=1)
            
            pred_modality = self.modality_predictors[i](others_concat)
            predicted_modalities.append(pred_modality)
        
        # Create enhanced representation with missing pattern information
        calibrated_features = self.calibration(
            torch.cat([features, missing_flags], dim=1)
        )
        
        # Apply contrastive projection
        contrastive_features = self.contrastive_projection(calibrated_features)
        
        # Reconstruct missing modalities
        reconstructed_modalities = []
        for i in range(self.num_modalities):
            is_present = missing_flags[:, i].bool()
            reconstructed = torch.zeros_like(modality_features[i])
            
            if is_present.any():
                reconstructed[is_present] = modality_features[i][is_present]
            
            is_missing = ~is_present
            if is_missing.any():
                pred = predicted_modalities[i]
                if pred.dtype != reconstructed.dtype:
                    pred = pred.to(reconstructed.dtype)
                reconstructed[is_missing] = pred[is_missing]
            
            reconstructed_modalities.append(reconstructed)
        
        return {
            'enhanced_features': calibrated_features,
            'predicted_modalities': predicted_modalities,
            'reconstructed_modalities': reconstructed_modalities,
            'contrastive_features': contrastive_features
        }


class CRLMMNAR(GeneralRecommender):
    """
    CRL-MMNAR model adapted for recommendation in MMG-benchmark.
    
    Core architecture components:
    1. Collaborative Filtering: User/Item ID embeddings + interaction graph GCN
    2. Multimodal Encoding: Image and text feature encoders
    3. Missing Pattern Encoding: MLP on missing flags -> representation δ
    4. Cross-Modal Attention: Z-gated attention with pattern awareness
    5. Graph Fusion: Item-modality bipartite graph enhancement
    6. Representation Balancing: Missing modality prediction & calibration
    
    This implementation adapts the clinical prediction model (multimodal_missingness.py)
    to work within the recommendation framework of MMG-benchmark, following patterns
    established by DGMRec.
    """

    def __init__(self, config, dataset):
        super(CRLMMNAR, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.n_ui_layers = config.get('n_ui_layers', 2)
        self.n_mm_layers = config.get('n_mm_layers', 2)
        self.knn_k = config.get('knn_k', 10)

        # Collaborative Filtering Model
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.n_nodes = self.n_users + self.n_items
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.norm_adj = self.norm_adj.to(self.device)
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        self.all_items = np.arange(self.n_items)
        self.complete_items = np.arange(self.n_items)
        self.missing_modal = config['missing_modal']
        if config['missing_modal']:
            self.preprocess_missing_modal(config)

        # Multimodal Item Features
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False).to(self.device)

            image_adj = build_sim(self.image_embedding.weight.detach())
            image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
            if self.missing_modal:
                image_adj[self.missing_items_v, :] = image_adj[:, self.missing_items_v] = 0.0
                image_adj[self.missing_items_v, self.missing_items_v] = 1.0
            self.image_adj = compute_normalized_laplacian(image_adj).to_sparse_coo()
            del image_adj

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False).to(self.device)

            text_adj = build_sim(self.text_embedding.weight.detach())
            text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
            if self.missing_modal:
                text_adj[self.missing_items_t, :] = text_adj[:, self.missing_items_t] = 0.0
                text_adj[self.missing_items_t, self.missing_items_t] = 1.0
            self.text_adj = compute_normalized_laplacian(text_adj).to_sparse_coo()
            del text_adj

        torch.cuda.empty_cache()

        # CRL-MMNAR specific hyperparameters
        self.alpha = config.get('alpha', 0.4)
        self.lambda_1 = config.get('lambda_1', 0.01)
        self.lambda_2 = config.get('lambda_2', 0.01)
        self.infoNCETemp = config.get('infoNCETemp', 0.4)
        self.alignBMTemp = config.get('alignBMTemp', 0.4)
        self.alignUITemp = config.get('alignUITemp', 0.2)
        self.num_attention_heads = config.get('num_attention_heads', 4)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.graph_hidden_dim = config.get('graph_hidden_dim', 32)

        # Modality encoders (adapted from original CRL-MMNAR)
        if self.v_feat is not None:
            self.img_encoder = ModalityEncoder(
                self.v_feat.shape[1], self.embedding_dim, dropout=self.dropout_rate
            ).to(self.device)
        
        if self.t_feat is not None:
            self.text_encoder = ModalityEncoder(
                self.t_feat.shape[1], self.embedding_dim, dropout=self.dropout_rate
            ).to(self.device)

        # Shared encoder for general features (like original)
        if self.v_feat is not None and self.t_feat is not None:
            self.shared_encoder_img = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
            self.shared_encoder_text = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
            nn.init.xavier_uniform_(self.shared_encoder_img.weight)
            nn.init.xavier_uniform_(self.shared_encoder_text.weight)

        # Specific encoders
        if self.v_feat is not None:
            self.img_encoder_s = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
            nn.init.xavier_uniform_(self.img_encoder_s.weight)
        
        if self.t_feat is not None:
            self.text_encoder_s = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
            nn.init.xavier_uniform_(self.text_encoder_s.weight)

        # Missing pattern embedding (δ -> z)
        self.missing_embedding = nn.Sequential(
            nn.Linear(2, self.embedding_dim),  # 2 modalities
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        ).to(self.device)

        # Cross-modal attention with z-gating
        self.cross_modal_attention = CrossModalAttention(
            self.embedding_dim,
            num_heads=self.num_attention_heads,
            dropout=self.dropout_rate
        ).to(self.device)

        # Graph-based modality fusion
        self.graph_fusion = GraphModalityFusion(
            self.embedding_dim,
            graph_hidden_dim=self.graph_hidden_dim,
            num_layers=self.n_mm_layers
        ).to(self.device)

        # Representation balancing module
        self.representation_balancing = RepresentationBalancingModule(
            hidden_dim=self.embedding_dim,
            num_modalities=2
        ).to(self.device)

        # Generators for missing modality (specific features)
        if self.v_feat is not None:
            self.img_gen = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            ).to(self.device)
            self.img_gen.apply(self._init_weight)
        
        if self.t_feat is not None:
            self.text_gen = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            ).to(self.device)
            self.text_gen.apply(self._init_weight)

        # Decoders for reconstruction
        if self.v_feat is not None and self.t_feat is not None:
            self.img_decoder = nn.Linear(self.embedding_dim * 2, self.v_feat.shape[1]).to(self.device)
            self.text_decoder = nn.Linear(self.embedding_dim * 2, self.t_feat.shape[1]).to(self.device)
            nn.init.xavier_uniform_(self.img_decoder.weight)
            nn.init.xavier_uniform_(self.text_decoder.weight)

        # User preference filters (for generating specific features)
        if self.v_feat is not None:
            self.img_preference = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False).to(self.device)
            nn.init.xavier_uniform_(self.img_preference.weight)
        
        if self.t_feat is not None:
            self.text_preference = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False).to(self.device)
            nn.init.xavier_uniform_(self.text_preference.weight)

        # Cross-modal generators (general features)
        if self.v_feat is not None and self.t_feat is not None:
            self.img2text = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            ).to(self.device)
            self.img2text.apply(self._init_weight)

            self.text2image = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            ).to(self.device)
            self.text2image.apply(self._init_weight)

        self.act_g = nn.Tanh()
        self.refresh_adj_counter = 0

    def _init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def init_mi_estimator(self):
        """Initialize mutual information estimator (required by trainer).
        
        CRLMMNAR doesn't use CLUB estimator like DGMRec, but we need to provide
        dummy attributes to be compatible with the training framework.
        """
        # Create dummy estimator modules for compatibility with train.py
        # These are not used in CRLMMNAR's loss calculation
        self.item_image_estimator = nn.Identity()
        self.item_text_estimator = nn.Identity()
        self.user_image_estimator = nn.Identity()
        self.user_text_estimator = nn.Identity()

    def preprocess_missing_modal(self, config):
        """Preprocess missing modal items (similar to DGMRec)."""
        possible_paths = [
            os.path.join('data', config['dataset']),
            os.path.abspath(config['data_path'] + config['dataset']),
            config['dataset'],
        ]

        dataset_path = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, config['dataset'] + '.inter')):
                dataset_path = path
                break

        if dataset_path is None:
            raise FileNotFoundError(f"Dataset directory not found. Tried:\n" +
                                  "\n".join(f"  - {p}" for p in possible_paths))

        print(f"[CRLMMNAR] Using dataset path: {dataset_path}")

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        self.missing_modality_type = config.get('missing_modality_type', 'all')

        modality_suffix_map = {
            'text': 'TEXT',
            't': 'TEXT',
            'image': 'IMAGE',
            'v': 'IMAGE',
            'visual': 'IMAGE',
            'all': ''
        }

        suffix = modality_suffix_map.get(self.missing_modality_type, '')

        if suffix:
            filename_with_suffix = f"missing_items_{self.missing_ratio}_{suffix}"
            filepath_with_suffix = os.path.join(dataset_path, f"{filename_with_suffix}.npy")

            if os.path.exists(filepath_with_suffix):
                self.missing_items = np.load(filepath_with_suffix, allow_pickle=True).item()
                print(f"[CRLMMNAR] Loaded missing items from: {filename_with_suffix}.npy")
            else:
                filename_default = f"missing_items_{self.missing_ratio}"
                filepath_default = os.path.join(dataset_path, f"{filename_default}.npy")

                if not os.path.exists(filepath_default):
                    raise FileNotFoundError(f"Missing modality file not found: tried {filepath_with_suffix} and {filepath_default}")

                self.missing_items = np.load(filepath_default, allow_pickle=True).item()
                print(f"[CRLMMNAR] Loaded missing items from: {filename_default}.npy (fallback)")
        else:
            filename = f"missing_items_{self.missing_ratio}"
            filepath = os.path.join(dataset_path, f"{filename}.npy")

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing modality file not found: {filepath}")

            self.missing_items = np.load(filepath, allow_pickle=True).item()
            print(f"[CRLMMNAR] Loaded missing items from: {filename}.npy")

        has_all_key = 'all' in self.missing_items and len(self.missing_items['all']) > 0

        if has_all_key:
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))
        else:
            self.missing_items_t = self.missing_items.get('t', np.array([]))
            self.missing_items_v = self.missing_items.get('v', np.array([]))

        self.complete_items = np.setdiff1d(
            np.arange(self.n_items), np.union1d(self.missing_items_v, self.missing_items_t)
        )

        non_missing_item_t = np.setdiff1d(self.all_items, self.missing_items_t)
        non_missing_item_v = np.setdiff1d(self.all_items, self.missing_items_v)

        if self.v_feat is not None:
            image_mean = self.v_feat[non_missing_item_v].mean(dim=0)
            self.v_feat[self.missing_items_v] = image_mean

        if self.t_feat is not None:
            text_mean = self.t_feat[non_missing_item_t].mean(dim=0)
            self.t_feat[self.missing_items_t] = text_mean

        print(f"[CRLMMNAR] Missing modality type: {self.missing_modality_type}")
        print(f"  - Items missing TEXT: {len(self.missing_items_t)}")
        print(f"  - Items missing IMAGE: {len(self.missing_items_v)}")
        print(f"  - Complete items (no missing): {len(self.complete_items)}")

    def pre_epoch_processing(self):
        """Called before each training epoch to generate missing modal features."""
        if not self.missing_modal:
            return
            
        # Get encoded features
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        # Generate missing modal features
        self.generate_missing_modal()

        self.refresh_adj_counter += 1
        if self.refresh_adj_counter % 5 == 0:
            self.update_adj()

    def generate_missing_modal(self):
        """Generate features for missing modalities (similar to DGMRec)."""
        if self.v_feat is None or self.t_feat is None:
            return
            
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        item_image_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.img_preference(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.text_preference(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]

        with torch.no_grad():
            # Generate general features using cross-modal generation
            item_text_g_gen, item_image_g_gen = self.img2text(item_image_g), self.text2image(item_text_g)
            
            # Apply modality graph layers
            for _ in range(self.n_mm_layers):
                item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
                item_text_g = torch.sparse.mm(self.text_adj, item_text_g)

            # Generate specific features from user preferences
            item_image_s, item_text_s = self.img_gen(item_image_filter), self.text_gen(item_text_filter)
            for _ in range(self.n_mm_layers):
                item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
                item_text_s = torch.sparse.mm(self.text_adj, item_text_s)

            # Reconstruct and update missing items
            item_image_recon = self.img_decoder(self.perturb(torch.cat([item_image_g, item_image_s], dim=1)))
            item_text_recon = self.text_decoder(self.perturb(torch.cat([item_text_g, item_text_s], dim=1)))

        with torch.no_grad():
            self.text_embedding.weight[self.missing_items['t']] = item_text_recon[self.missing_items['t']]
            self.image_embedding.weight[self.missing_items['v']] = item_image_recon[self.missing_items['v']]

    def update_adj(self):
        """Update adjacency matrices after generating missing features."""
        if self.v_feat is None or self.t_feat is None:
            return
            
        with torch.no_grad():
            t_index = self.missing_items_t
            v_index = self.missing_items_v

        with torch.no_grad():
            self.image_adj = self.image_adj.cpu().to_dense()
            torch.cuda.empty_cache()

            image_adj = build_sim(self.image_embedding.weight.detach())
            image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
            image_adj = compute_normalized_laplacian(image_adj).cpu()

            self.image_adj[v_index] = image_adj[v_index] * self.alpha + self.image_adj[v_index] * (1 - self.alpha)
            self.image_adj = self.image_adj.to_sparse_coo()
            del image_adj

            self.text_adj = self.text_adj.cpu().to_dense()
            torch.cuda.empty_cache()

            text_adj = build_sim(self.text_embedding.weight.detach())
            text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
            text_adj = compute_normalized_laplacian(text_adj).cpu()

            self.text_adj[t_index] = text_adj[t_index] * self.alpha + self.text_adj[t_index] * (1 - self.alpha)
            self.text_adj = self.text_adj.to_sparse_coo()
            del text_adj

            torch.cuda.empty_cache()
            self.image_adj = self.image_adj.to(self.device)
            self.text_adj = self.text_adj.to(self.device)

    def cge(self, user_emb, item_emb, adj):
        """Collaborative Graph Embedding."""
        ego_embeddings = torch.cat((user_emb, item_emb), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        user_embeddings, item_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        del ego_embeddings, side_embeddings

        return user_embeddings, item_embedding

    def mge(self):
        """Multimodal Graph Embedding - encode image and text features."""
        if self.v_feat is not None:
            item_image_g = F.sigmoid(self.shared_encoder_img(self.act_g(self.img_encoder(self.image_embedding.weight))))
            item_image_s = F.sigmoid(self.img_encoder_s(self.image_embedding.weight))
        else:
            item_image_g = None
            item_image_s = None

        if self.t_feat is not None:
            item_text_g = F.sigmoid(self.shared_encoder_text(self.act_g(self.text_encoder(self.text_embedding.weight))))
            item_text_s = F.sigmoid(self.text_encoder_s(self.text_embedding.weight))
        else:
            item_text_g = None
            item_text_s = None

        return item_image_g, item_text_g, item_image_s, item_text_s

    def get_item_multimodal_features(self, item_ids=None):
        """
        Get fused multimodal features for items using CRL-MMNAR architecture.
        
        This is the core adaptation that brings CRL-MMNAR's fusion strategy
        into the recommendation framework.
        """
        if item_ids is None:
            item_ids = torch.arange(self.n_items).to(self.device)
        
        # Get basic encoded features
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()
        
        # Create missing flags for items (batch_size x 2)
        # For simplicity, assume all items have both modalities initially
        # Missing items will have been filled with mean values
        batch_size = len(item_ids)
        missing_flags = torch.ones(batch_size, 2).to(self.device)
        
        # Check which items are actually missing each modality
        if hasattr(self, 'missing_items'):
            v_set = set(self.missing_items_v.tolist()) if len(self.missing_items_v) > 0 else set()
            t_set = set(self.missing_items_t.tolist()) if len(self.missing_items_t) > 0 else set()
            
            for i, idx in enumerate(item_ids):
                idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
                if idx_val in v_set:
                    missing_flags[i, 1] = 0  # image missing
                if idx_val in t_set:
                    missing_flags[i, 0] = 0  # text missing
        
        # Compute missing pattern IDs (0-3 for 2 modalities)
        pattern_ids = torch.sum(
            missing_flags * (2 ** torch.arange(2, device=self.device)),
            dim=1
        ).long()
        
        # Get missing representation z = MLP(δ)
        missing_repr = self.missing_embedding(missing_flags)  # [B, D]
        
        # Gather modality features for requested items
        if item_image_g is not None:
            img_feats_g = item_image_g[item_ids]
            img_feats_s = item_image_s[item_ids]
        else:
            img_feats_g = torch.zeros(batch_size, self.embedding_dim).to(self.device)
            img_feats_s = torch.zeros(batch_size, self.embedding_dim).to(self.device)
        
        if item_text_g is not None:
            text_feats_g = item_text_g[item_ids]
            text_feats_s = item_text_s[item_ids]
        else:
            text_feats_g = torch.zeros(batch_size, self.embedding_dim).to(self.device)
            text_feats_s = torch.zeros(batch_size, self.embedding_dim).to(self.device)
        
        # Use general features for cross-modal attention
        modality_features = [img_feats_g, text_feats_g]
        
        # Apply cross-modal attention with z-gating
        attended_features, _, _ = self.cross_modal_attention(
            modality_features, missing_flags, pattern_ids, missing_repr
        )
        
        # Apply graph-based fusion
        graph_enhanced = self.graph_fusion(attended_features, missing_flags)
        
        # Merge features (initial + attended + graph-enhanced)
        initial_combined = (img_feats_g + text_feats_g) / 2.0
        enhanced_features = (initial_combined + attended_features + graph_enhanced) / 3.0
        
        # Apply representation balancing
        rb_output = self.representation_balancing(
            enhanced_features, 
            [img_feats_s, text_feats_s], 
            missing_flags
        )
        
        final_features = rb_output['enhanced_features']
        
        return final_features

    def calculate_loss(self, interaction):
        """
        Calculate training loss combining BPR loss with CRL-MMNAR auxiliary losses.
        """
        users, pos_items, neg_items = interaction

        # 1. Collaborative Graph Embedding
        user_embeddings, item_id_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )
        
        # 2. Multimodal Graph Embedding
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        all_items, _ = torch.unique(
            torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False
        )

        # Index for finding Non-missing Items (Recon/Gen Loss)
        t_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_t)
        v_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_v)
        tv_index = np.setdiff1d(
            all_items.detach().cpu().numpy(), np.union1d(self.missing_items_t, self.missing_items_v)
        )

        # InfoNCE loss for cross-modal alignment (if both modalities exist)
        loss_InfoNCE = torch.tensor(0.0).to(self.device)
        if item_image_g is not None and item_text_g is not None and len(tv_index) > 0:
            loss_InfoNCE = self.InfoNCE(item_image_g[tv_index], item_text_g[tv_index], temperature=self.infoNCETemp)

        # User preference filtering
        item_image_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.img_preference(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.text_preference(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]

        # Filter general features
        if item_image_g is not None:
            item_image_g_filtered = torch.einsum("ij, ij -> ij", item_image_filter, item_image_g)
        if item_text_g is not None:
            item_text_g_filtered = torch.einsum("ij, ij -> ij", item_text_filter, item_text_filter)

        # Item-Item Graph GCN (General)
        for _ in range(self.n_mm_layers):
            if item_image_g is not None:
                item_image_g_filtered = torch.sparse.mm(self.image_adj, item_image_g_filtered)
            if item_text_g is not None:
                item_text_g_filtered = torch.sparse.mm(self.text_adj, item_text_g_filtered)
        
        if item_image_g is not None:
            user_image_g = torch.sparse.mm(self.adj, item_image_g_filtered) * self.num_inters[:self.n_users]
        if item_text_g is not None:
            user_text_g = torch.sparse.mm(self.adj, item_text_g_filtered) * self.num_inters[:self.n_users]

        # Generation loss (generate specific features from filtered preferences)
        loss_gen = torch.tensor(0.0).to(self.device)
        if item_image_g is not None and item_text_g is not None:
            item_image_g_gen = self.text2image(self.perturb(item_text_g_filtered))
            item_text_g_gen = self.img2text(self.perturb(item_image_g_filtered))
            
            item_image_s_gen = self.img_gen(self.perturb(item_image_filter))
            item_text_s_gen = self.text_gen(self.perturb(item_text_filter))

            if len(v_index) > 0:
                loss_gen += MSELoss(item_image_s[v_index], item_image_s_gen[v_index])
            if len(t_index) > 0:
                loss_gen += MSELoss(item_text_s[t_index], item_text_s_gen[t_index])
            if len(tv_index) > 0:
                loss_gen += MSELoss(item_text_g[tv_index], item_text_g_gen[tv_index])
                loss_gen += MSELoss(item_image_g[tv_index], item_image_g_gen[tv_index])

        # Filter specific features
        if item_image_s is not None:
            item_image_s_filtered = torch.einsum("ij, ij -> ij", item_image_filter, item_image_s)
        if item_text_s is not None:
            item_text_s_filtered = torch.einsum("ij, ij -> ij", item_text_filter, item_text_s)

        # Item-Item Graph GCN (Specific)
        for _ in range(self.n_mm_layers):
            if item_image_s is not None:
                item_image_s_filtered = torch.sparse.mm(self.image_adj, item_image_s_filtered)
            if item_text_s is not None:
                item_text_s_filtered = torch.sparse.mm(self.text_adj, item_text_s_filtered)
        
        if item_image_s is not None:
            user_image_s = torch.sparse.mm(self.adj, item_image_s_filtered) * self.num_inters[:self.n_users]
        if item_text_s is not None:
            user_text_s = torch.sparse.mm(self.adj, item_text_s_filtered) * self.num_inters[:self.n_users]

        # Combine multimodal features for final representation
        if item_image_g is not None and item_text_g is not None:
            image_embs = torch.concat([user_image_g + user_image_s, item_image_g + item_image_s], dim=0)
            text_embs = torch.concat([user_text_g + user_text_s, item_text_g + item_text_s], dim=0)
            _, item_image_final = torch.split(image_embs, [self.n_users, self.n_items], dim=0)
            _, item_text_final = torch.split(text_embs, [self.n_users, self.n_items], dim=0)
        elif item_image_g is not None:
            item_image_final = item_image_g + item_image_s
            item_text_final = None
        elif item_text_g is not None:
            item_text_final = item_text_g + item_text_s
            item_image_final = None
        else:
            item_image_final = None
            item_text_final = None

        # Alignment losses
        loss_alignUI = torch.tensor(0.0).to(self.device)
        loss_alignBM = torch.tensor(0.0).to(self.device)
        
        if item_image_g is not None and item_text_g is not None:
            # User-Item alignment
            loss_alignUI = self.InfoNCE(
                user_embeddings[users], item_id_embedding[pos_items], temperature=self.alignUITemp
            )
            # Multimodal-user alignment
            loss_alignUI += self.InfoNCE(
                user_image_g[users] + user_text_g[users],
                item_image_g[pos_items] + item_text_g[pos_items],
                temperature=self.infoNCETemp
            )
            # Item-multimodal alignment
            loss_alignBM = self.InfoNCE(
                item_id_embedding[pos_items], 
                (item_image_g[pos_items] + item_text_g[pos_items]) if item_image_g is not None else item_text_g[pos_items],
                temperature=self.alignBMTemp
            )
            loss_alignBM += self.InfoNCE(
                user_embeddings[users],
                (user_image_g[users] + user_text_g[users]) if user_image_g is not None else user_text_g[users],
                temperature=self.alignBMTemp
            )

        # Fuse embeddings for final prediction
        if item_image_g is not None and item_text_g is not None:
            user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
            item_emb = item_id_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3
        elif item_image_g is not None:
            user_emb = user_embeddings + (user_image_g + user_image_s) / 2
            item_emb = item_id_embedding + (item_image_g + item_image_s) / 2
        elif item_text_g is not None:
            user_emb = user_embeddings + (user_text_g + user_text_s) / 2
            item_emb = item_id_embedding + (item_text_g + item_text_s) / 2
        else:
            user_emb = user_embeddings
            item_emb = item_id_embedding

        user_emb, pos_item_emb, neg_item_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]

        # Main BPR loss
        loss_main_bpr = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        # Regularization loss
        reg_components = [user_emb, pos_item_emb, neg_item_emb, user_embeddings[users], item_id_embedding[pos_items]]
        if item_image_final is not None:
            reg_components.append(item_image_final[pos_items])
        if item_text_final is not None:
            reg_components.append(item_text_final[pos_items])
        loss_reg = self.calculate_reg_loss(*reg_components)

        # Reconstruction loss
        loss_recon = torch.tensor(0.0).to(self.device)
        if item_image_g is not None and item_text_g is not None:
            loss_recon = self.calculate_recon_loss(
                torch.cat([item_image_g, item_image_s], dim=1),
                torch.cat([item_text_g, item_text_s], dim=1)
            )

        # Combine all losses
        loss_disentangle = self.lambda_1 * loss_InfoNCE
        loss_generation = loss_gen + loss_recon
        loss_align = self.lambda_2 * (loss_alignUI + loss_alignBM)

        total_loss = loss_main_bpr + loss_disentangle + loss_generation + loss_align + loss_reg

        return total_loss

    def full_sort_predict(self, interaction):
        """
        Full sort prediction for evaluation.
        Returns scores for all items given users.
        """
        users, _ = interaction

        # 1. Collaborative Graph Embedding
        user_embeddings, item_id_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )
        
        # 2. Multimodal Graph Embedding
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        # User preference filtering
        if self.v_feat is not None:
            item_image_filter = torch.sparse.mm(
                self.adj.t(), F.tanh(self.img_preference(self.user_embedding.weight))
            ) * self.num_inters[self.n_users:]
        if self.t_feat is not None:
            item_text_filter = torch.sparse.mm(
                self.adj.t(), F.tanh(self.text_preference(self.user_embedding.weight))
            ) * self.num_inters[self.n_users:]

        # Filter and propagate general features
        if item_image_g is not None:
            item_image_g_filtered = torch.einsum("ij, ij -> ij", item_image_filter, item_image_g)
            for _ in range(self.n_mm_layers):
                item_image_g_filtered = torch.sparse.mm(self.image_adj, item_image_g_filtered)
            user_image_g = torch.sparse.mm(self.adj, item_image_g_filtered) * self.num_inters[:self.n_users]
        
        if item_text_g is not None:
            item_text_g_filtered = torch.einsum("ij, ij -> ij", item_text_filter, item_text_g)
            for _ in range(self.n_mm_layers):
                item_text_g_filtered = torch.sparse.mm(self.text_adj, item_text_g_filtered)
            user_text_g = torch.sparse.mm(self.adj, item_text_g_filtered) * self.num_inters[:self.n_users:]

        # Filter and propagate specific features
        if item_image_s is not None:
            item_image_s_filtered = torch.einsum("ij, ij -> ij", item_image_filter, item_image_s)
            for _ in range(self.n_mm_layers):
                item_image_s_filtered = torch.sparse.mm(self.image_adj, item_image_s_filtered)
            user_image_s = torch.sparse.mm(self.adj, item_image_s_filtered) * self.num_inters[:self.n_users]
        
        if item_text_s is not None:
            item_text_s_filtered = torch.einsum("ij, ij -> ij", item_text_filter, item_text_s)
            for _ in range(self.n_mm_layers):
                item_text_s_filtered = torch.sparse.mm(self.text_adj, item_text_s_filtered)
            user_text_s = torch.sparse.mm(self.adj, item_text_s_filtered) * self.num_inters[:self.n_users:]

        # Fuse Features
        if item_image_g is not None and item_text_g is not None:
            user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
            item_emb = item_id_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3
        elif item_image_g is not None:
            user_emb = user_embeddings + (user_image_g + user_image_s) / 2
            item_emb = item_id_embedding + (item_image_g + item_image_s) / 2
        elif item_text_g is not None:
            user_emb = user_embeddings + (user_text_g + user_text_s) / 2
            item_emb = item_id_embedding + (item_text_g + item_text_s) / 2
        else:
            user_emb = user_embeddings
            item_emb = item_id_embedding

        user_emb, pos_item_emb = user_emb[users], item_emb

        score = user_emb @ pos_item_emb.T
        return score

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        import scipy.sparse as sp
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(
            zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz
        ))
        data_dict.update(dict(zip(
            zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M.nnz
        )))
        for key, value in data_dict.items():
            A[key] = value
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return sumArr, torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def calculate_reg_loss(self, *embs):
        loss_reg = torch.tensor(0.0).to(self.device)
        for emb in embs:
            loss_reg += torch.norm(emb, p=2)
        loss_reg /= embs[-1].shape[0]
        return loss_reg * 1e-5

    def calculate_recon_loss(self, image, text):
        if self.v_feat is None or self.t_feat is None:
            return torch.tensor(0.0).to(self.device)
        item_image_recon = self.img_decoder(self.perturb(image.detach()))
        item_text_recon = self.text_decoder(self.perturb(text.detach()))

        loss = 0
        loss += F.mse_loss(item_image_recon, self.image_embedding.weight) * 0.1
        loss += F.mse_loss(item_text_recon, self.text_embedding.weight) * 0.1
        return loss

    def bpr_loss(self, users, pos_items, neg_items):
        if len(pos_items.shape) == 2:
            pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
            neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        else:
            pos_scores = torch.einsum("ik, ijk -> ij", users, pos_items)
            neg_scores = torch.einsum("ik, ijk -> ij", users, neg_items)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def InfoNCE(self, view1, view2, temperature=0.4):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def forward(self):
        pass

    def perturb(self, x):
        noise = torch.rand_like(x).to(self.device)
        x = x + torch.sign(x) * F.normalize(noise, dim=-1) * 0.1
        return x
