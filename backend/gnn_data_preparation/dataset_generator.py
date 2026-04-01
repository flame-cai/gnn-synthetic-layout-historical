# data_creation/dataset_generator.py
import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from typing import List, Dict

class HistoricalLayoutGNNDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for the historical layout analysis task.
    """
    def __init__(self, root, transform=None, pre_transform=None, data_list=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # We process manually, so this can be empty
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Data is assumed to be present locally in the specified raw dir
        pass

    def process(self):
        # If data is provided directly, just save it
        if self.data_list is not None:
            data, slices = self.collate(self.data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            # This would be used if we were processing from raw files inside this class
            # But we are doing it in the main script for better control over folds.
            print("Processing from raw files is not implemented directly in the class. "
                  "Please provide a data_list during initialization.")

def create_sklearn_dataframe(data_list: List[Data], page_map: Dict[int, str], config) -> pd.DataFrame:
    """
    Converts a list of PyG Data objects into a single pandas DataFrame
    suitable for training with scikit-learn models.
    """
    if not config.sklearn_format.enabled:
        return None

    all_edges_data = []
    
    node_feature_names = []
    if config.features.use_node_coordinates: node_feature_names.extend(['x', 'y'])
    if config.features.use_node_font_size: node_feature_names.append('font_size')
    # Add other node features names here if they are simple values
    
    for i, data in enumerate(data_list):
        page_id = page_map[i]
        edge_index = data.edge_index.T.numpy()
        edge_y = data.edge_y.numpy()
        
        for edge_idx, (u, v) in enumerate(edge_index):
            row = {'page_id': page_id, 'source_node_id': u, 'target_node_id': v}
            
            # Source and Target Node Features
            if "source_node_features" in config.sklearn_format.features:
                for feat_idx, name in enumerate(node_feature_names):
                    row[f'source_{name}'] = data.x[u, feat_idx].item()
            
            if "target_node_features" in config.sklearn_format.features:
                for feat_idx, name in enumerate(node_feature_names):
                    row[f'target_{name}'] = data.x[v, feat_idx].item()
            
            # Edge Features
            if "edge_features" in config.sklearn_format.features and data.edge_attr is not None:
                # This needs to be more specific based on config, but for now we dump them
                for feat_idx in range(data.edge_attr.shape[1]):
                    row[f'edge_attr_{feat_idx}'] = data.edge_attr[edge_idx, feat_idx].item()
                    
            # Page Features
            if "page_features" in config.sklearn_format.features and hasattr(data, 'page_aspect_ratio'):
                row['page_aspect_ratio'] = data.page_aspect_ratio.item()

            # Target Label
            row['label'] = edge_y[edge_idx]
            
            all_edges_data.append(row)

    df = pd.DataFrame(all_edges_data)
    # Note: N-hop features are complex and would require a separate graph traversal step here.
    # This is a placeholder for that advanced functionality.
    if config.sklearn_format.use_n_hop_features:
        print("Warning: N-hop feature generation for sklearn is an advanced feature and not yet implemented.")
        
    return df