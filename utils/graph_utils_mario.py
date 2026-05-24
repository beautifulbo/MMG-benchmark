# coding: utf-8

"""
Mario Graph Utilities
图结构相关工具函数: 距离矩阵计算、DGL 图构建等
"""
import os
import torch
import numpy as np

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


def compute_shortest_path_distance_matrix(graph, max_distance=6):
    """
    计算 DGL 图的全局最短路径距离矩阵

    Args:
        graph: DGL 图对象
        max_distance: 最大距离截断值（超过此值的距离设为 max_distance）

    Returns:
        dist_matrix: (num_nodes, num_nodes) 的 LongTensor 距离矩阵
    """
    if not DGL_AVAILABLE:
        raise ImportError("DGL is required for distance matrix computation")

    num_nodes = graph.num_nodes()

    if NX_AVAILABLE and num_nodes <= 10000:
        return _compute_with_networkx(graph, num_nodes, max_distance)
    else:
        return _compute_with_bfs(graph, num_nodes, max_distance)


def _compute_with_networkx(graph, num_nodes, max_distance):
    """使用 NetworkX 计算精确最短路径距离矩阵"""
    nx_graph = dgl.to_networkx(graph).to_undirected()
    dist_matrix = torch.full((num_nodes, num_nodes), max_distance, dtype=torch.long)

    for node_id in range(num_nodes):
        try:
            lengths = dict(nx.single_source_shortest_path_length(nx_graph, node_id, cutoff=max_distance))
            for target, length in lengths.items():
                if target < num_nodes:
                    dist_matrix[node_id, target] = min(int(length), max_distance)
        except Exception:
            pass

    return dist_matrix


def _compute_with_bfs(graph, num_nodes, max_distance):
    """使用 BFS 近似计算距离矩阵（适用于大图）"""
    dist_matrix = torch.full((num_nodes, num_nodes), max_distance, dtype=torch.long)

    for i in range(num_nodes):
        dist_matrix[i, i] = 0

    for hop in range(1, max_distance + 1):
        if hop == 1:
            current_reachable = {i: set() for i in range(num_nodes)}
            src, dst = graph.edges()
            for s, d in zip(src.numpy(), dst.numpy()):
                current_reachable[s].add(d)
                current_reachable[d].add(s)
        else:
            new_reachable = {i: set() for i in range(num_nodes)}
            for node_id in range(num_nodes):
                for neighbor in list(current_reachable.get(node_id, set()))[:1000]:
                    for nn in list(current_reachable.get(neighbor, set()))[:500]:
                        if nn != node_id and dist_matrix[node_id, nn] >= max_distance:
                            new_reachable[node_id].add(nn)
                            dist_matrix[node_id, nn] = min(hop, max_distance)
                            dist_matrix[nn, node_id] = min(hop, max_distance)
            current_reachable = new_reachable

    return dist_matrix


def build_dgl_graph_from_interaction(interaction_matrix, n_users, n_items, add_self_loop=False):
    """
    从交互矩阵构建 DGL 同构图/异构图

    Args:
        interaction_matrix: scipy.sparse COO 格式的交互矩阵 (n_users × n_items)
        n_users: 用户数量
        n_items: 物品数量
        add_self_loop: 是否添加自环

    Returns:
        graph: DGL 图对象
    """
    if not DGL_AVAILABLE:
        raise ImportError("DGL is required for graph construction")

    rows = interaction_matrix.row
    cols = interaction_matrix.col

    user_item_edges = []
    item_user_edges = []

    for r, c in zip(rows, cols):
        user_item_edges.append((r, n_users + c))
        item_user_edges.append((n_users + c, r))

    all_src = [e[0] for e in user_item_edges] + [e[0] for e in item_user_edges]
    all_dst = [e[1] for e in user_item_edges] + [e[1] for e in item_user_edges]

    total_nodes = n_users + n_items
    graph = dgl.graph((all_src, all_dst), num_nodes=total_nodes)

    if add_self_loop:
        # DGL 不直接支持自环，需要特殊处理
        pass

    return graph


def build_homogeneous_graph_from_features(n_nodes, feature_dim=None, device='cpu'):
    """
    构建一个空的同构图（用于纯节点特征场景）

    Args:
        n_nodes: 节点数
        feature_dim: 特征维度（可选）
        device: 设备

    Returns:
        graph: DGL 图对象
    """
    if not DGL_AVAILABLE:
        raise ImportError("DGL is required")

    graph = dgl.graph(([], []), num_nodes=n_nodes)
    return graph


def load_or_compute_dist_matrix(config, graph, cache_dir=None):
    """
    加载或计算并缓存距离矩阵

    Args:
        config: 配置字典，包含 dist_matrix_path 等
        graph: DGL 图对象
        cache_dir: 缓存目录（默认为 config['checkpoint_dir']）

    Returns:
        dist_matrix: 距离矩阵张量
    """
    dist_matrix_path = config.get('dist_matrix_path')

    if dist_matrix_path and os.path.exists(dist_matrix_path):
        print(f"[GraphUtils] Loading pre-computed distance matrix from {dist_matrix_path}")
        return torch.load(dist_matrix_path, mmap=True, weights_only=True)

    if cache_dir is None:
        cache_dir = config.get('checkpoint_dir', 'saved')

    cache_path = os.path.join(cache_dir, 'mario_dist_matrix.pt')
    if os.path.exists(cache_path):
        print(f"[GraphUtils] Loading cached distance matrix from {cache_path}")
        return torch.load(cache_path, mmap=True, weights_only=True)

    print("[GraphUtils] Computing shortest path distance matrix...")
    buckets_num = config.get('buckets_num', 6)
    dist_matrix = compute_shortest_path_distance_matrix(graph, max_distance=buckets_num - 1)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(dist_matrix, cache_path)
    print(f"[GraphUtils] Distance matrix cached to {cache_path} (shape: {dist_matrix.shape})")

    return dist_matrix


def get_node_neighbors(graph, node_id, hop=1, max_neighbors=None):
    """
    获取节点的 hop 邻居

    Args:
        graph: DGL 图对象
        node_id: 节点 ID
        hop: 跳数 (1 或 2)
        max_neighbors: 最大邻居数限制

    Returns:
        neighbors: 邻居 ID 列表
    """
    try:
        succ = graph.successors(node_id) if hasattr(graph, 'successors') else graph.successors(node_id)
        if hop == 1:
            neighbors = [s.item() if isinstance(s, torch.Tensor) else s for s in succ]
        else:
            hop2_set = set()
            for n in succ:
                n_item = n.item() if isinstance(n, torch.Tensor) else n
                for h in graph.successors(n_item):
                    h_item = h.item() if isinstance(h, torch.Tensor) else h
                    if h_item != node_id:
                        hop2_set.add(h_item)
            neighbors = list(hop2_set)
    except Exception as e:
        print(f"[GraphUtils] Warning: Failed to get neighbors for node {node_id}: {e}")
        neighbors = []

    if max_neighbors and len(neighbors) > max_neighbors:
        neighbors = neighbors[:max_neighbors]

    return neighbors


def get_topk_neighbors_by_similarity(center_feat, neighbor_feats, k=5):
    """
    基于余弦相似度选择 Top-K 邻居

    Args:
        center_feat: (dim,) 中心节点特征
        neighbor_feats: (N, dim) 候选邻居特征
        k: 选择数量

    Returns:
        topk_indices: Top-K 邻居索引
    """
    import torch.nn.functional as F

    center_feat = center_feat.unsqueeze(0) if center_feat.dim() == 1 else center_feat
    sim_scores = F.cosine_similarity(center_feat, neighbor_feats, dim=-1)

    k = min(k, len(sim_scores))
    _, topk_indices = torch.topk(sim_scores, k)

    return topk_indices
