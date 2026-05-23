# coding: utf-8

"""
DGMRec: Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.abstract_recommender import GeneralRecommender
from .base.loss import MSELoss
from .base.mi_estimator import CLUBSample
from data.utils.graph_utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood


class DGMRec(GeneralRecommender):
    """
    DGMRec model for missing modality scenarios.
    Disentangles modality features into general and specific components,
    then generates missing modality features by integrating aligned features.
    """

    def __init__(self, config, dataset):
        super(DGMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.n_ui_layers = config['n_ui_layers']
        self.n_mm_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']

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

        # Multimodal Item Feature
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
        self.loss_nce = config['loss_nce']

        # Encoder / Decoder / Preference
        self.image_encoder = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        self.text_encoder = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        self.shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder.weight)
        nn.init.xavier_uniform_(self.text_encoder.weight)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        self.image_encoder_s = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        self.text_encoder_s = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder_s.weight)
        nn.init.xavier_uniform_(self.text_encoder_s.weight)

        self.image_preference_ = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.text_preference_ = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.image_preference_.weight)
        nn.init.xavier_uniform_(self.text_preference_.weight)

        self.image_decoder = nn.Linear(self.embedding_dim * 2, self.v_feat.shape[1]).to(self.device)
        self.text_decoder = nn.Linear(self.embedding_dim * 2, self.t_feat.shape[1]).to(self.device)
        nn.init.xavier_uniform_(self.image_decoder.weight)
        nn.init.xavier_uniform_(self.text_decoder.weight)

        # Generator for Specific Feature
        self.image_gen = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.image_gen.apply(self.init_weight)

        self.text_gen = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.text_gen.apply(self.init_weight)

        # Generator for General Feature
        self.image2text = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.image2text.apply(self.init_weight)

        self.text2image = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.text2image.apply(self.init_weight)

        # Hyper-parameter
        self.alpha = config['alpha']
        self.lambda_1 = config['lambda_1']
        self.lambda_2 = config['lambda_2']

        self.infoNCETemp = config['infoNCETemp']
        self.alignBMTemp = config['alignBMTemp']
        self.alignUITemp = config['alignUITemp']

        self.act_g = nn.Tanh()
        self.refresh_adj_counter = 0

    def init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def init_mi_estimator(self):
        self.item_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()
        self.user_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()
        self.item_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()
        self.user_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()

        params = (list(self.item_image_estimator.parameters()) +
                  list(self.user_image_estimator.parameters()) +
                  list(self.item_text_estimator.parameters()) +
                  list(self.user_text_estimator.parameters()))

        self.optimizer_club = torch.optim.Adam(params, lr=1e-4)

    def preprocess_missing_modal(self, config):
        # Try multiple possible dataset paths
        possible_paths = [
            os.path.join('data', config['dataset']),                    # data/baby/
            os.path.abspath(config['data_path'] + config['dataset']),   # ../data/baby/
            config['dataset'],                                           # baby/
        ]

        dataset_path = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, config['dataset'] + '.inter')):
                dataset_path = path
                break

        if dataset_path is None:
            raise FileNotFoundError(
                f"Dataset directory not found. Tried:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                f"\nPlease ensure the dataset is in the correct location."
            )

        print(f"[DGMRec] Using dataset path: {dataset_path}")

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        self.missing_modality_type = config.get('missing_modality_type', 'all')

        # Try to load file with modality suffix first, then fall back to original filename
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
            # New format: try with modality suffix first
            filename_with_suffix = f"missing_items_{self.missing_ratio}_{suffix}"
            filepath_with_suffix = os.path.join(dataset_path, f"{filename_with_suffix}.npy")

            if os.path.exists(filepath_with_suffix):
                self.missing_items = np.load(filepath_with_suffix, allow_pickle=True).item()
                print(f"[DGMRec] Loaded missing items from: {filename_with_suffix}.npy")
            else:
                # Fall back to original format if new file doesn't exist
                print(f"[DGMRec] Warning: {filename_with_suffix}.npy not found, falling back to default")
                filename_default = f"missing_items_{self.missing_ratio}"
                filepath_default = os.path.join(dataset_path, f"{filename_default}.npy")

                if not os.path.exists(filepath_default):
                    raise FileNotFoundError(
                        f"Missing modality file not found: tried {filepath_with_suffix} and {filepath_default}"
                    )

                self.missing_items = np.load(filepath_default, allow_pickle=True).item()
                print(f"[DGMRec] Loaded missing items from: {filename_default}.npy (fallback)")
        else:
            # Original format (all modalities or backward compatibility)
            filename = f"missing_items_{self.missing_ratio}"
            filepath = os.path.join(dataset_path, f"{filename}.npy")

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing modality file not found: {filepath}")

            self.missing_items = np.load(filepath, allow_pickle=True).item()
            print(f"[DGMRec] Loaded missing items from: {filename}.npy")

        # Compute missing item indices - compatible with both old and new formats
        has_all_key = 'all' in self.missing_items and len(self.missing_items['all']) > 0

        if has_all_key:
            # Old format or all-modality mode: combine 'all' with specific modalities
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))
        else:
            # New single-modality format: use specific modality only
            self.missing_items_t = self.missing_items.get('t', np.array([]))
            self.missing_items_v = self.missing_items.get('v', np.array([]))

        self.complete_items = np.setdiff1d(
            np.arange(self.n_items), np.union1d(self.missing_items_v, self.missing_items_t)
        )

        self.items_tv = np.setdiff1d(
            np.arange(self.n_items), np.union1d(self.missing_items_t, self.missing_items_v)
        )

        non_missing_item_t = np.setdiff1d(self.all_items, self.missing_items_t)
        non_missing_item_v = np.setdiff1d(self.all_items, self.missing_items_v)

        image_mean = self.v_feat[non_missing_item_v].mean(dim=0)
        text_mean = self.t_feat[non_missing_item_t].mean(dim=0)

        self.v_feat[self.missing_items_v] = image_mean
        self.t_feat[self.missing_items_t] = text_mean

        print(f"[DGMRec] Missing modality type: {self.missing_modality_type}")
        print(f"  - Items missing TEXT: {len(self.missing_items_t)}")
        print(f"  - Items missing IMAGE: {len(self.missing_items_v)}")
        print(f"  - Complete items (no missing): {len(self.complete_items)}")

    def pre_epoch_processing(self):
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        total_loss_mi = 0.0
        for _ in range(5):
            self.item_image_estimator.train()
            self.item_text_estimator.train()

            item_rand_idx = torch.randperm(self.n_items)[:2048]

            loss_mi = 0.0
            loss_mi += self.item_image_estimator.learning_loss(item_image_s[item_rand_idx], item_image_g[item_rand_idx])
            loss_mi += self.item_text_estimator.learning_loss(item_text_s[item_rand_idx], item_text_g[item_rand_idx])

            self.optimizer_club.zero_grad()
            loss_mi.backward(retain_graph=True)
            self.optimizer_club.step()
            total_loss_mi += loss_mi.detach().item()

        self.item_image_estimator.eval()
        self.item_text_estimator.eval()

        self.refresh_adj_counter += 1
        self.generate_missing_modal()
        if self.refresh_adj_counter % 5 == 0:
            self.update_adj()

    def generate_missing_modal(self):
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()
        item_image_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.image_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.text_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]

        with torch.no_grad():
            item_text_g, item_image_g = self.image2text(item_image_g), self.text2image(item_text_g)
            for _ in range(self.n_mm_layers):
                item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
                item_text_g = torch.sparse.mm(self.text_adj, item_text_g)

            item_image_s, item_text_s = self.image_gen(item_image_filter), self.text_gen(item_text_filter)
            for _ in range(self.n_mm_layers):
                item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
                item_text_s = torch.sparse.mm(self.text_adj, item_text_s)

            item_image_recon = self.image_decoder(self.perturb(torch.concat([item_image_g, item_image_s], dim=1)))
            item_text_recon = self.text_decoder(self.perturb(torch.concat([item_text_g, item_text_s], dim=1)))

        with torch.no_grad():
            self.text_embedding.weight[self.missing_items['t']] = item_text_recon[self.missing_items['t']]
            self.image_embedding.weight[self.missing_items['v']] = item_image_recon[self.missing_items['v']]

    def update_adj(self):
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
        """Modality Graph Embedding."""
        item_image_g = F.sigmoid(self.shared_encoder(self.act_g(self.image_encoder(self.image_embedding.weight))))
        item_text_g = F.sigmoid(self.shared_encoder(self.act_g(self.text_encoder(self.text_embedding.weight))))

        item_image_s = F.sigmoid(self.image_encoder_s(self.image_embedding.weight))
        item_text_s = F.sigmoid(self.text_encoder_s(self.text_embedding.weight))
        return item_image_g, item_text_g, item_image_s, item_text_s

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction

        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        all_items, _ = torch.unique(
            torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False
        )

        # index for finding Non-missing Items (Recon/Gen Loss)
        t_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_t)
        v_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_v)
        tv_index = np.setdiff1d(
            all_items.detach().cpu().numpy(), np.union1d(self.missing_items_t, self.missing_items_v)
        )

        loss_InfoNCE = self.InfoNCE(item_image_g[tv_index], item_text_g[tv_index], temperature=self.infoNCETemp)

        item_image_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.image_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.text_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]

        # Filtering (General)
        item_image_g = torch.einsum("ij, ij -> ij", item_image_filter, item_image_g)
        item_text_g = torch.einsum("ij, ij -> ij", item_text_filter, item_text_g)

        # Item-Item Graph GCN (General)
        for _ in range(self.n_mm_layers):
            item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
            item_text_g = torch.sparse.mm(self.text_adj, item_text_g)
        user_image_g = torch.sparse.mm(self.adj, item_image_g) * self.num_inters[:self.n_users]
        user_text_g = torch.sparse.mm(self.adj, item_text_g) * self.num_inters[:self.n_users]

        # Loss_gen
        item_image_g_gen, item_text_g_gen = self.text2image(self.perturb(item_text_g)), self.image2text(self.perturb(item_image_g))
        item_image_s_gen = self.image_gen(self.perturb(item_image_filter))
        item_text_s_gen = self.text_gen(self.perturb(item_text_filter))

        loss_gen = 0
        loss_gen += MSELoss(item_image_s[v_index], item_image_s_gen[v_index])
        loss_gen += MSELoss(item_text_s[t_index], item_text_s_gen[t_index])
        loss_gen += MSELoss(item_text_g[tv_index], item_text_g_gen[tv_index])
        loss_gen += MSELoss(item_image_g[tv_index], item_image_g_gen[tv_index])

        # Filtering (Specific)
        item_image_s = torch.einsum("ij, ij -> ij", item_image_filter, item_image_s)
        item_text_s = torch.einsum("ij, ij -> ij", item_text_filter, item_text_s)

        # Item-Item Graph GCN (Specific)
        for _ in range(self.n_mm_layers):
            item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
            item_text_s = torch.sparse.mm(self.text_adj, item_text_s)
        user_image_s = torch.sparse.mm(self.adj, item_image_s) * self.num_inters[:self.n_users]
        user_text_s = torch.sparse.mm(self.adj, item_text_s) * self.num_inters[:self.n_users]

        image_embs = torch.concat([user_image_g + user_image_s, item_image_g + item_image_s], dim=0)
        text_embs = torch.concat([user_text_g + user_text_s, item_text_g + item_text_s], dim=0)

        _, item_image_final = torch.split(image_embs, [self.n_users, self.n_items], dim=0)
        _, item_text_final = torch.split(text_embs, [self.n_users, self.n_items], dim=0)

        # MI Sampler Loss
        loss_club = 0.0
        loss_club += self.item_image_estimator(item_image_s, item_image_g)
        loss_club += self.item_text_estimator(item_text_s, item_text_g)

        loss_InfoNCE += self.InfoNCE(
            user_image_g[users], user_text_g[users], temperature=self.infoNCETemp
        )

        loss_alignUI = self.InfoNCE(
            user_embeddings[users], item_embedding[pos_items], temperature=self.alignUITemp
        )
        loss_alignUI += self.InfoNCE(
            user_image_g[users] + user_text_g[users],
            item_image_g[pos_items] + item_text_g[pos_items],
            temperature=self.infoNCETemp
        )
        loss_alignUI += self.InfoNCE(
            user_image_s[users], item_image_s[pos_items], temperature=self.alignUITemp
        )
        loss_alignUI += self.InfoNCE(
            user_text_s[users], item_text_s[pos_items], temperature=self.alignUITemp
        )

        loss_alignBM = self.InfoNCE(
            item_embedding[pos_items], item_image_g[pos_items] + item_text_g[pos_items],
            temperature=self.alignBMTemp
        )
        loss_alignBM += self.InfoNCE(
            user_embeddings[users], user_image_g[users] + user_text_g[users],
            temperature=self.alignBMTemp
        )

        user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
        item_emb = item_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3

        user_emb, pos_item_emb, neg_item_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]

        loss_main_bpr = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        loss_reg = self.calculate_reg_loss(
            user_embeddings[users], item_embedding[pos_items], item_embedding[neg_items],
            item_image_final[pos_items], item_text_final[pos_items]
        )

        image_final = torch.concat([item_image_g, item_image_s], dim=1)
        text_final = torch.concat([item_text_g, item_text_s], dim=1)
        loss_recon = self.calculate_recon_loss(image_final, text_final)

        loss_disentangle = self.lambda_1 * (loss_club + loss_InfoNCE)
        loss_generation = loss_gen + loss_recon
        loss_align = self.lambda_2 * (loss_alignUI + loss_alignBM)

        return loss_main_bpr + loss_disentangle + loss_generation + loss_align + loss_reg

    def full_sort_predict(self, interaction):
        users, _ = interaction

        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        # Filtering (General)
        item_image_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.image_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.text_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]

        item_image_g = torch.einsum("ij, ij -> ij", item_image_filter, item_image_g)
        item_text_g = torch.einsum("ij, ij -> ij", item_text_filter, item_text_g)

        # Item-Item Graph GCN (General)
        for _ in range(self.n_mm_layers):
            item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
            item_text_g = torch.sparse.mm(self.text_adj, item_text_g)
        user_image_g = torch.sparse.mm(self.adj, item_image_g) * self.num_inters[:self.n_users]
        user_text_g = torch.sparse.mm(self.adj, item_text_g) * self.num_inters[:self.n_users]

        # Filtering (Specific)
        item_image_s = torch.einsum("ij, ij -> ij", item_image_filter, item_image_s)
        item_text_s = torch.einsum("ij, ij -> ij", item_text_filter, item_text_s)

        # Item-Item Graph GCN (Specific)
        for _ in range(self.n_mm_layers):
            item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
            item_text_s = torch.sparse.mm(self.text_adj, item_text_s)
        user_image_s = torch.sparse.mm(self.adj, item_image_s) * self.num_inters[:self.n_users]
        user_text_s = torch.sparse.mm(self.adj, item_text_s) * self.num_inters[:self.n_users]

        # Fuse Features
        user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
        item_emb = item_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3

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
            zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz
        )))
        for key, value in data_dict.items():
            A[key] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return sumArr, torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def calculate_reg_loss(self, user_emb, pos_items_emb, neg_item_emb, image_emb, text_emb):
        loss_reg = self.reg_loss(user_emb, pos_items_emb, neg_item_emb) * 1e-5
        loss_reg += self.reg_loss(image_emb) * 0.1
        loss_reg += self.reg_loss(text_emb) * 0.1
        return loss_reg

    def calculate_recon_loss(self, image, text):
        item_image_recon = self.image_decoder(self.perturb(image.detach()))
        item_text_recon = self.text_decoder(self.perturb(text.detach()))

        loss = 0
        loss += F.mse_loss(item_image_recon, self.image_embedding.weight) * 0.1
        loss += F.mse_loss(item_text_recon, self.text_embedding.weight) * 0.1
        return loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

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