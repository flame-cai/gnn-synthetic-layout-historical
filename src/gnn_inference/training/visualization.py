# training/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D # For manual legend

def visualize_graph_predictions(data, predictions, output_path: Path):
    """
    Visualizes the ground truth vs. predicted graph for a single page.
    - Faint Grey: All unique edges from the input graph.
    - Green: True Positive edges
    - Red: False Positive edges
    - Orange Dashed: False Negative edges
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pos = data.x[:, :2].cpu().numpy() # Node positions
    
    # Create copies to avoid modifying the original data object's tensors
    edge_index = data.edge_index.cpu().numpy().copy()
    y_true = data.edge_y.cpu().numpy().copy()
    y_pred = predictions.copy()
    
    # --- START OF MINIMAL, ROBUST FIX ---

    # We need a version of edge_index just for the background plot
    edge_index_for_background = edge_index.copy()

    # Gracefully check for bidirectionality to avoid double-plotting
    is_bidirectional = False
    num_edges = edge_index.shape[1]
    if num_edges > 0 and num_edges % 2 == 0:
        num_half_edges = num_edges // 2
        first_half = edge_index[:, :num_half_edges]
        second_half_flipped = np.flip(edge_index[:, num_half_edges:], axis=0)
        if np.array_equal(first_half, second_half_flipped):
            is_bidirectional = True

    # If the graph is bidirectional, we only need to process the first half
    # for plotting the colored (TP, FP, FN) edges.
    if is_bidirectional:
        num_undirected_edges = edge_index.shape[1] // 2
        edge_index = edge_index[:, :num_undirected_edges]
        y_true = y_true[:num_undirected_edges]
        y_pred = y_pred[:num_undirected_edges]
        
        # Also trim the background graph to avoid double plotting it
        edge_index_for_background = edge_index_for_background[:, :num_undirected_edges]
    
    # If the graph is unidirectional, the `if` block is skipped, and we use the full arrays.
    
    # --- END OF FIX ---

    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    plt.figure(figsize=(18, 14))
    plt.scatter(pos[:, 0], -pos[:, 1], s=10, c='black', zorder=5) # Invert y-axis
    plt.gca().set_aspect('equal', adjustable='box')


    def draw_edges(edges, color, linestyle='-', linewidth=1.0, alpha=1.0, zorder=1):
        # Your original, working draw_edges function
        if edges.shape[1] == 0:
            return
        for u, v in edges.T:
            plt.plot([pos[u, 0], pos[v, 0]], [-pos[u, 1], -pos[v, 1]], 
                     color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    # 1. Draw the background graph
    draw_edges(edge_index_for_background, 'lightgrey', linewidth=0.5, zorder=1)
    
    # 2. Draw the categorized edges
    if np.any(tp_mask):
        draw_edges(edge_index[:, tp_mask], 'green', linewidth=1.5, zorder=3)
    if np.any(fp_mask):
        draw_edges(edge_index[:, fp_mask], 'red', linewidth=1.0, zorder=2)
    if np.any(fn_mask):
        draw_edges(edge_index[:, fn_mask], 'orange', linestyle='--', linewidth=1.5, zorder=4)

    # 3. Create a manual legend
    legend_elements = [
        Line2D([0], [0], color='lightgrey', lw=1, label='Input Graph Edge'),
        Line2D([0], [0], color='green', lw=2, label='Correctly Kept (TP)'),
        Line2D([0], [0], color='red', lw=1.5, label='Incorrectly Kept (FP)'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Incorrectly Deleted (FN)'),
        plt.scatter([], [], s=30, color='black', label='Character (Node)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='large')

    plt.title(f"Prediction Visualization for Page: {data.page_id}", fontsize=16)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_inference_predictions(data, predictions, output_path: Path):
    """
    Visualizes the model's predictions on new data where no ground truth is available.
    - Faint Grey: All edges from the input graph.
    - Blue: Edges the model predicted as "keep".
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pos = data.x[:, :2].cpu().numpy()
    edge_index = data.edge_index.cpu().numpy().copy()
    y_pred = predictions.copy()
    
    # Gracefully handle uni- or bi-directional graphs to avoid double-plotting
    is_bidirectional = False
    num_edges = edge_index.shape[1]
    if num_edges > 0 and num_edges % 2 == 0:
        num_half_edges = num_edges // 2
        if np.array_equal(edge_index[:, :num_half_edges], np.flip(edge_index[:, num_half_edges:], axis=0)):
            is_bidirectional = True
    
    if is_bidirectional:
        num_undirected_edges = edge_index.shape[1] // 2
        edge_index = edge_index[:, :num_undirected_edges]
        y_pred = y_pred[:num_undirected_edges]
    
    pred_pos_mask = (y_pred == 1)

    plt.figure(figsize=(18, 14))
    plt.scatter(pos[:, 0], -pos[:, 1], s=10, c='black', zorder=5)
    plt.gca().set_aspect('equal', adjustable='box')

    def draw_edges(edges, color, linestyle='-', linewidth=1.0, alpha=1.0, zorder=1):
        if edges.shape[1] == 0:
            return
        for u, v in edges.T:
            plt.plot([pos[u, 0], pos[v, 0]], [-pos[u, 1], -pos[v, 1]], 
                     color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)

    # 1. Draw the full input graph
    draw_edges(edge_index, 'lightgrey', linewidth=0.5, zorder=1)
    
    # 2. Draw the kept edges on top
    if np.any(pred_pos_mask):
        draw_edges(edge_index[:, pred_pos_mask], 'blue', linewidth=1.5, zorder=3)

    # 3. Create a manual legend
    legend_elements = [
        Line2D([0], [0], color='lightgrey', lw=1, label='Input Graph Edge'),
        Line2D([0], [0], color='blue', lw=2, label='Predicted "Keep" Edge'),
        plt.scatter([], [], s=30, color='black', label='Character (Node)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='large')

    plt.title(f"Inference Visualization for Page: {data.page_id}", fontsize=16)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

# --- END OF NEW FUNCTION ---