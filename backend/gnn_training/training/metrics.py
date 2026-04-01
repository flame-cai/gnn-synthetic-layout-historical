# training/metrics.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import comb
from scipy.sparse import coo_matrix


# (Your rand_index placeholder function)
def rand_index(true_labels, pred_labels):
    return 0.0

def calculate_single_graph_metrics(y_true: np.ndarray, y_pred: np.ndarray, data, true_textline_labels_for_page=None):
    """
    Calculates a dictionary of metrics for a SINGLE graph.
    
    Args:
        y_true: Ground truth binary labels for the graph's edges.
        y_pred: Predicted binary labels for the graph's edges.
        data: The PyG Data object for the single graph.
        true_textline_labels_for_page: Ground truth textline labels for this page's nodes.
    """
    metrics = {}
    
    # --- Simplified GED Calculation for one graph ---
    edge_index = data.edge_index.cpu().numpy()
    
    true_pos_mask = (y_true == 1)
    pred_pos_mask = (y_pred == 1)
    
    gt_edges = set(map(tuple, edge_index[:, true_pos_mask].T))
    pred_edges = set(map(tuple, edge_index[:, pred_pos_mask].T))
    
    gt_edges_undirected = {tuple(sorted(e)) for e in gt_edges}
    pred_edges_undirected = {tuple(sorted(e)) for e in pred_edges}
    
    false_positives = len(pred_edges_undirected - gt_edges_undirected)
    false_negatives = len(gt_edges_undirected - pred_edges_undirected)
    metrics['simplified_ged'] = false_positives + false_negatives

    # --- Rand Index Calculation for one graph ---
    if true_textline_labels_for_page is not None:
        if not pred_edges_undirected:
            pred_clusters = np.arange(data.num_nodes)
        else:
            pred_adj = csr_matrix((np.ones(len(pred_edges_undirected)), 
                                  list(zip(*pred_edges_undirected))),
                                 shape=(data.num_nodes, data.num_nodes))
            _, pred_clusters = connected_components(csgraph=pred_adj, directed=False, return_labels=True)
        
        metrics['rand_index'] = rand_index(true_textline_labels_for_page, pred_clusters)
            
    return metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, data=None, true_textline_labels=None):
    """
    Calculates a dictionary of metrics for edge classification.
    
    Args:
        y_true: Ground truth binary labels for edges.
        y_pred: Predicted binary labels for edges.
        data: The PyG Data object for graph-level metrics.
        true_textline_labels: Ground truth textline labels for nodes (for Rand Index).
    """
    metrics = {}
    
    # Standard classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # Graph-level metrics
    if data is not None:
        num_nodes = data.num_nodes
        edge_index = data.edge_index.cpu().numpy()
        
        true_pos_mask = (y_true == 1)
        pred_pos_mask = (y_pred == 1)
        
        gt_edges = set(map(tuple, edge_index[:, true_pos_mask].T))
        pred_edges = set(map(tuple, edge_index[:, pred_pos_mask].T))
        
        gt_edges_undirected = {tuple(sorted(e)) for e in gt_edges}
        pred_edges_undirected = {tuple(sorted(e)) for e in pred_edges}
        
        false_positives = len(pred_edges_undirected - gt_edges_undirected)
        false_negatives = len(gt_edges_undirected - pred_edges_undirected)
        metrics['simplified_ged'] = false_positives + false_negatives
        
        # Rand Index
        if true_textline_labels is not None:
            if not pred_edges_undirected:
                # If no edges are predicted, every node is its own cluster.
                pred_clusters = np.arange(num_nodes)
            else:
                pred_adj = csr_matrix((np.ones(len(pred_edges_undirected)), 
                                      list(zip(*pred_edges_undirected))), # Using list() for robustness
                                     shape=(num_nodes, num_nodes))
                _, pred_clusters = connected_components(csgraph=pred_adj, directed=False, return_labels=True)
            
            metrics['rand_index'] = rand_index(true_textline_labels, pred_clusters)
            
    return metrics




def get_textlines_from_edges(num_nodes, edge_index, edge_labels):
    """
    Identifies text lines as connected components from a given set of edges.
    A text line is a set of node indices.

    Args:
        num_nodes (int): The total number of nodes in the graph.
        edge_index (torch.Tensor or np.ndarray): The edge index tensor of shape [2, num_edges].
        edge_labels (torch.Tensor or np.ndarray): The binary labels (0 or 1) for each edge.

    Returns:
        list[set[int]]: A list of sets, where each set contains the node indices of a text line.
    """
    # Ensure tensors are on the CPU and are numpy arrays for scipy
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    if isinstance(edge_labels, torch.Tensor):
        edge_labels = edge_labels.cpu().numpy()
        
    # Filter for edges that are predicted to be part of a text line (label == 1)
    positive_edges = edge_index[:, edge_labels == 1]
    
    # If there are no positive edges, every node is its own isolated component (a line of one char)
    if positive_edges.shape[1] == 0:
        return [{i} for i in range(num_nodes)]

    # Create a sparse adjacency matrix for efficient connected component analysis
    adj_matrix = coo_matrix(
        (np.ones(positive_edges.shape[1]), (positive_edges[0], positive_edges[1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Find the connected components, which represent the predicted text lines
    n_components, labels = connected_components(
        csgraph=adj_matrix, directed=False, return_labels=True
    )
    
    # Group nodes by their component ID to form the final text lines
    components = [set() for _ in range(n_components)]
    for node_idx, component_id in enumerate(labels):
        components[component_id].add(node_idx)
        
    return [c for c in components if c] # Return only non-empty components

def calculate_textline_level_counts(ground_truth_lines, predicted_lines, iou_threshold=0.5):
    """
    Calculates object-level TP, FP, and FN for text lines using node-based IoU matching.
    This function implements the core logic of our mAP@0.5 equivalent.

    Args:
        ground_truth_lines (list[set[int]]): A list of ground-truth text lines.
        predicted_lines (list[set[int]]): A list of predicted text lines.
        iou_threshold (float): The IoU threshold to consider a match a True Positive.

    Returns:
        tuple[int, int, int]: The number of True Positives, False Positives, and False Negatives.
    """
    num_gt = len(ground_truth_lines)
    num_pred = len(predicted_lines)

    # Handle edge cases where one or both sets are empty
    if num_gt == 0 and num_pred == 0: return 0, 0, 0
    if num_pred == 0: return 0, 0, num_gt  # All ground truth lines were missed
    if num_gt == 0: return 0, num_pred, 0 # All predicted lines are hallucinations

    # Create an IoU matrix: rows are GT lines, columns are predicted lines
    iou_matrix = np.zeros((num_gt, num_pred))
    for i, gt_line in enumerate(ground_truth_lines):
        for j, pred_line in enumerate(predicted_lines):
            intersection = len(gt_line.intersection(pred_line))
            union = len(gt_line.union(pred_line))
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    # Greedily match predicted lines to ground truth lines based on highest IoU
    # This ensures a one-to-one mapping for counting TP, FP, FN
    matches = []
    # Find all potential matches that are above the threshold
    for i in range(num_gt):
        for j in range(num_pred):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))  # Store (iou, gt_idx, pred_idx)

    # Sort matches by IoU score in descending order to prioritize the best matches
    matches.sort(key=lambda x: x[0], reverse=True)
    
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    for _, gt_idx, pred_idx in matches:
        # If this GT and this Pred haven't been matched yet, form a match
        if gt_idx not in matched_gt_indices and pred_idx not in matched_pred_indices:
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
            
    tp = len(matched_gt_indices)
    fp = num_pred - tp
    fn = num_gt - tp
    
    return tp, fp, fn
