# data_creation/feature_engineering.py
import torch
import numpy as np

try:
    from .config_models import FeaturesConfig
except ImportError:
    from config_models import FeaturesConfig


def get_node_features(points: np.ndarray, heuristic_degrees: np.ndarray, config: FeaturesConfig) -> torch.Tensor:
    """Combines all enabled node features into a single tensor."""
    features_list = []
    
    if config.use_node_coordinates:
        features_list.append(torch.from_numpy(points[:, :2]).float())
    
    if config.use_node_font_size:
        features_list.append(torch.from_numpy(points[:, 2]).float().unsqueeze(1))
        
    if config.use_heuristic_degree:
        deg_feat = encode_categorical_feature(
            heuristic_degrees,
            config.heuristic_degree_encoding,
            config.heuristic_degree_encoding_params.linear_map_factor,
            config.heuristic_degree_encoding_params.one_hot_max_degree
        )
        features_list.append(deg_feat)
        
    return torch.cat(features_list, dim=1)


def get_edge_features(edge_index: torch.Tensor, node_features: torch.Tensor, heuristic_edge_counts: dict, config: FeaturesConfig) -> torch.Tensor:
    """Computes all enabled edge features for the given edges."""
    features_list = []
    source_nodes, target_nodes = edge_index[0], edge_index[1]
    
    # Use original node coordinates for distance calculations
    # Assuming first 2 features are x, y
    source_pos = node_features[source_nodes, :2]
    target_pos = node_features[target_nodes, :2]
    
    relative_pos = target_pos - source_pos
    
    if config.use_relative_distance:
        features_list.append(relative_pos) # rel_x, rel_y
        
    if config.use_euclidean_distance:
        dist = torch.linalg.norm(relative_pos, dim=1).unsqueeze(1)
        features_list.append(dist)
        
    if config.use_aspect_ratio_rel:
        # Add epsilon for stability
        aspect = torch.abs(relative_pos[:, 1]) / (torch.abs(relative_pos[:, 0]) + 1e-6)
        features_list.append(aspect.unsqueeze(1))
        
    if config.use_overlap:
        overlaps = []
        for u, v in edge_index.T.numpy():
            edge_key = tuple(sorted((u, v)))
            overlaps.append(heuristic_edge_counts.get(edge_key, 0))
        overlaps = np.array(overlaps)
        
        overlap_feat = encode_categorical_feature(
            overlaps,
            config.overlap_encoding,
            config.overlap_encoding_params.linear_map_factor,
            config.overlap_encoding_params.one_hot_max_overlap
        )
        features_list.append(overlap_feat)

    if not features_list:
        return None

    return torch.cat(features_list, dim=1)

def encode_categorical_feature(values: np.ndarray, method: str, factor: float, max_val: int) -> torch.Tensor:
    """Encodes a categorical feature using the specified method."""
    if method == "linear_map":
        return torch.from_numpy(values).float().unsqueeze(1) * factor
    elif method == "one_hot":
        # Clamp values to the max to avoid oversized tensors
        values_clamped = np.clip(values, 0, max_val)
        one_hot = torch.nn.functional.one_hot(torch.from_numpy(values_clamped).long(), num_classes=max_val + 1)
        return one_hot.float()
    else:
        raise ValueError(f"Unknown encoding method: {method}")