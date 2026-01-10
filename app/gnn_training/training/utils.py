# training/utils.py
import torch
import random
import numpy as np
import os
from pathlib import Path
import logging
import pandas as pd
import torch.nn.functional as F

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_config: str) -> torch.device:
    """Gets the torch device based on config and availability, and logs the choice."""
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}") # This line confirms the choice in your logs
    return device

def save_checkpoint(model, optimizer, epoch, metrics, path: Path):
    """Saves a model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model,
        # 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    logging.info(f"Saved checkpoint to {path}")

def calculate_class_weights(dataset) -> torch.Tensor:
    """Calculates class weights for imbalanced datasets."""
    all_labels = torch.cat([data.edge_y for data in dataset])
    num_total = len(all_labels)
    num_positives = all_labels.sum().item()
    num_negatives = num_total - num_positives
    
    if num_positives == 0 or num_negatives == 0:
        return torch.tensor([1.0, 1.0])
        
    # Weight for class 0 (negative) and class 1 (positive)
    # weight = total_samples / (n_classes * n_samples_per_class)
    weight_0 = num_total / (2 * num_negatives)
    weight_1 = num_total / (2 * num_positives)
    
    return torch.tensor([weight_0, weight_1])

def create_prediction_log(data_list, predictions, labels, model_name: str, fold_idx: int) -> pd.DataFrame:
    """Creates a detailed CSV log of predictions for analysis."""
    rows = []
    current_edge_idx = 0
    for data in data_list:
        num_edges = data.edge_index.shape[1]
        page_preds = predictions[current_edge_idx : current_edge_idx + num_edges]
        page_labels = labels[current_edge_idx : current_edge_idx + num_edges]
        
        for i in range(num_edges):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            row = {
                'model': model_name,
                'fold': fold_idx,
                'page_id': data.page_id,
                'source_node_id': u,
                'target_node_id': v,
                'prediction': page_preds[i],
                'label': page_labels[i],
            }
            # Add node and edge features for context
            for feat_idx in range(data.x.shape[1]):
                row[f'source_node_feat_{feat_idx}'] = data.x[u, feat_idx].item()
                row[f'target_node_feat_{feat_idx}'] = data.x[v, feat_idx].item()
            if data.edge_attr is not None:
                for feat_idx in range(data.edge_attr.shape[1]):
                    row[f'edge_feat_{feat_idx}'] = data.edge_attr[i, feat_idx].item()
            
            rows.append(row)
        
        current_edge_idx += num_edges
        
    return pd.DataFrame(rows)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss