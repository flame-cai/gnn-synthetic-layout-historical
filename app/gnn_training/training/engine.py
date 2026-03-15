# training/engine.py
import torch
from torch_geometric.loader import DataLoader # Changed from torch.utils.data
from tqdm import tqdm
import numpy as np
import pandas as pd # To handle results accumulation
import logging # Import logging
#from .metrics import calculate_single_graph_metrics # Import the new function
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .metrics import calculate_single_graph_metrics, get_textlines_from_edges, calculate_textline_level_counts



def train_one_epoch(model, dataloader: DataLoader, optimizer, loss_fn, device: torch.device):
    """
    Trains the model for one epoch in a memory-efficient manner.

    This function is optimized to prevent out-of-memory errors by:
    1. Calculating accuracy in a streaming fashion, batch by batch.
    2. Only accumulating predictions and labels required for metrics like F1-score,
       which cannot be calculated iteratively.
    """
    model.train()
    
    # Initialize accumulators for metrics
    total_loss = 0
    total_correct_preds = 0
    total_edges = 0
    
    # Lists to store labels and predictions for metrics that require the full dataset (e.g., F1)
    all_preds_list = []
    all_labels_list = []

    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = loss_fn(logits, batch.edge_y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # --- Metrics Calculation (Batch-level) ---
        
        # Update total loss
        total_loss += loss.item() * batch.num_graphs
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        labels = batch.edge_y
        
        # Update streaming accuracy metrics
        total_correct_preds += (preds == labels).sum().item()
        total_edges += labels.numel() # Use numel() for total number of elements
        
        # Append tensors to lists for later concatenation.
        # This is more memory-efficient than converting to numpy inside the loop.
        all_preds_list.append(preds.cpu())
        all_labels_list.append(labels.cpu())

    # --- Aggregate Metrics (Epoch-level) ---
    
    # Concatenate all predictions and labels once after the loop
    all_preds = torch.cat(all_preds_list).numpy()
    all_labels = torch.cat(all_labels_list).numpy()
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct_preds / total_edges
    
    # Calculate precision, recall, and F1-score
    # These metrics require all data, hence the concatenation above.
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, loss_fn, device: torch.device):
    """
    Evaluates a GNN model, calculating both edge-level and object-level (textline) metrics.
    """
    model.eval()
    
    all_edge_preds, all_edge_labels = [], []
    total_loss = 0

    # --- START OF CHANGE: Add accumulators for our new object-level textline metric ---
    total_textline_tp = 0
    total_textline_fp = 0
    total_textline_fn = 0
    # --- END OF CHANGE ---

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)
        
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = loss_fn(logits, batch.edge_y)
        total_loss += loss.item() * batch.num_graphs
        
        batch_preds = torch.argmax(logits, dim=1)
        
        # Store all edge-level predictions and labels for overall metrics
        all_edge_preds.append(batch_preds.cpu().numpy())
        all_edge_labels.append(batch.edge_y.cpu().numpy())
        
        # --- START OF CHANGE: Calculate and accumulate textline metrics for each graph ---
        # This logic is robust and relies on batch_size=1 for evaluation, which is standard practice.
        if batch.num_graphs == 1:
            # 1. Identify the ground-truth "objects" (text lines)
            gt_lines = get_textlines_from_edges(batch.num_nodes, batch.edge_index, batch.edge_y)
            
            # 2. Identify the predicted "objects" (text lines)
            pred_lines = get_textlines_from_edges(batch.num_nodes, batch.edge_index, batch_preds)
            
            # 3. Compare them to get TP, FP, FN for this graph
            tp, fp, fn = calculate_textline_level_counts(gt_lines, pred_lines, iou_threshold=0.5)
            
            # 4. Accumulate counts over the whole dataset
            total_textline_tp += tp
            total_textline_fp += fp
            total_textline_fn += fn
        else:
            # Log a warning if batch size is not 1, as textline metric will be skipped.
            logging.warning(
                f"Batch size is {batch.num_graphs} during evaluation. "
                "Textline F1-Score calculation is skipped for this batch. "
                "Please use batch_size=1 for accurate validation metrics."
            )
        # --- END OF CHANGE ---

    # --- Aggregate Metrics ---

    # 1. Aggregate edge-level metrics
    all_edge_preds = np.concatenate(all_edge_preds)
    all_edge_labels = np.concatenate(all_edge_labels)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_edge_labels, all_edge_preds, average='macro', zero_division=0
    )

    final_metrics = {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': accuracy_score(all_edge_labels, all_edge_preds),
        'f1_score_macro': f1_macro # Keep your primary edge-level metric
    }
    
    # --- START OF CHANGE: Calculate final Textline F1-Score and add to metrics dict ---
    # Calculate precision and recall from the accumulated TP, FP, FN counts
    precision_denom = total_textline_tp + total_textline_fp
    recall_denom = total_textline_tp + total_textline_fn
    
    textline_precision = total_textline_tp / precision_denom if precision_denom > 0 else 0.0
    textline_recall = total_textline_tp / recall_denom if recall_denom > 0 else 0.0
    
    # Calculate the final F1-Score
    f1_denom = textline_precision + textline_recall
    textline_f1 = 2 * (textline_precision * textline_recall) / f1_denom if f1_denom > 0 else 0.0

    # Add the new, more meaningful metrics to the dictionary that gets logged and used for early stopping
    final_metrics['textline_f1_score'] = textline_f1
    final_metrics['textline_precision'] = textline_precision
    final_metrics['textline_recall'] = textline_recall
    
    logging.info(f"Textline Metrics | TP: {total_textline_tp}, FP: {total_textline_fp}, FN: {total_textline_fn} | F1: {textline_f1:.4f}")
    # --- END OF CHANGE ---

    # Note: The logic for 'graph_metrics_list' from your original code was removed
    # as it was not being populated. This new implementation is cleaner.

    return final_metrics, all_edge_preds, all_edge_labels