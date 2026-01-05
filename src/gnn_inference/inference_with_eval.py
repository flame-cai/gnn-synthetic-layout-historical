# inference_with_eval.py
#
# Standalone script to run and evaluate the FULL model pipeline (Input Graph + GNN).
# This script performs a system-level evaluation where the "universe" of edges is
# the union of the input graph and the ground truth MST graph. It correctly
# accounts for and visualizes edges missed by the input graph creation step.

import torch
import numpy as np
import yaml
import logging
from pathlib import Path
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import comb
import torch.nn.functional as F
from training.metrics import get_textlines_from_edges, calculate_textline_level_counts

import matplotlib
matplotlib.use('Agg')  # MUST be called before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shutil
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from segment_from_point_clusters import segmentLinesFromPointClusters


# --- START: Imports from your project structure ---
from gnn_data_preparation.config_models import DatasetCreationConfig, GroundTruthConfig
from gnn_data_preparation.graph_constructor import create_input_graph_edges, create_ground_truth_graph_edges
from gnn_data_preparation.feature_engineering import get_node_features, get_edge_features
from gnn_data_preparation.utils import setup_logging
from torch_geometric.data import Data
from training.utils import get_device
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# --- END: Imports from your project structure ---

import os
import sys
# Get the directory where the current script is located (gnn_inference)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)
# Add 'src' to the system path so Python can find 'gnn_training'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print(f"Added {parent_dir} to sys.path to allow imports from 'gnn_training'")


# ===================================================================
#           UTILITY, METRIC, AND VISUALIZATION FUNCTIONS
# ===================================================================

def fit_robust_line_and_extend(points: np.ndarray, extend_percentage: float = 0.05, robust_method: str = 'huber'):
    """
    Fits a robust line to a set of 2D points, extends it, and returns the new endpoints.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 2) with the [x, y] coordinates.
        extend_percentage (float): The percentage to extend the line by on each end.
        robust_method (str): The robust regression method to use ('huber' or 'ransac').

    Returns:
        tuple: A tuple containing two points, ((x1, y1), (x2, y2)), representing the
               start and end of the extended best-fit line.
    """
    if len(points) < 2:
        return None  # Cannot fit a line to less than two points

    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    # 1. Fit a robust regression model
    if robust_method.lower() == 'ransac':
        # RANSAC is excellent for significant outliers but computationally more expensive.
        model = RANSACRegressor(min_samples=2, residual_threshold=5.0, max_trials=100)
    elif robust_method.lower() == 'huber':
        # Huber is a good default, less sensitive to outliers than OLS.
        model = HuberRegressor(epsilon=1.35)
    else:
        raise ValueError("robust_method must be either 'ransac' or 'huber'")

    try:
        model.fit(x, y)
        y_pred = model.predict(x)
    except Exception:
        return None # Could not fit a model

    # 2. Determine the endpoints of the fitted line on the original data range
    x_min, x_max = np.min(x), np.max(x)
    y_min_pred = model.predict([[x_min]])[0]
    y_max_pred = model.predict([[x_max]])[0]

    p1 = np.array([x_min, y_min_pred])
    p2 = np.array([x_max, y_max_pred])

    # 3. Extend the line by the specified percentage
    direction_vector = p2 - p1
    line_length = np.linalg.norm(direction_vector)
    
    if line_length == 0:
      return ( (p1[0], p1[1]), (p2[0],p2[1]) )

    unit_vector = direction_vector / line_length

    # Calculate the new endpoints
    p1_extended = p1 - unit_vector * (line_length * extend_percentage)
    p2_extended = p2 + unit_vector * (line_length * extend_percentage)

    return ((p1_extended[0], p1_extended[1]), (p2_extended[0], p2_extended[1]))

def find_connected_components(positive_edges: set, num_nodes: int) -> list[list[int]]:
    """
    Finds all connected components (groups of nodes) in the graph.
    This version is guaranteed to be stateless and work correctly in a loop.
    """
    # --- THIS IS THE FIX ---
    # All state variables are defined here, inside the function call,
    # ensuring they are brand new for every page.
    adj = defaultdict(list)
    for u, v in positive_edges:
        adj[u].append(v)
        adj[v].append(u)

    components = []
    visited = set()
    # --- END FIX ---

    if not positive_edges:
        return [[i] for i in range(num_nodes)]

    for i in range(num_nodes):
        if i not in visited:
            component = []
            q = [i]
            visited.add(i)
            while q:
                u = q.pop(0)
                component.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            components.append(component)
            
    return components

def trace_component_with_backtracking(component: list[int], adj: defaultdict) -> list[int]:
    """
    Traces a single, continuous path that covers every edge of a component using a
    clean, standard, non-recursive DFS algorithm. This is guaranteed to terminate and is
    robust to any graph structure, including those with cycles.
    """
    if not component:
        return []

    visited_edges = set()
    path = []

    # A good starting point is a leaf node (degree 1) if one exists.
    start_node = component[0]
    for node in component:
        # We need to check if the node is actually in the adjacency list,
        # as a component could be a single isolated node.
        if node in adj and len(adj[node]) == 1:
            start_node = node
            break

    # Handle the edge case of a single, isolated node with no edges.
    if not adj.get(start_node):
        return [start_node]

    stack = [start_node]
    path.append(start_node)

    while stack:
        u = stack[-1]  # Peek at the top of the stack

        # Find the next unvisited neighbor to travel to.
        next_neighbor = None
        # Sort neighbors for a consistent traversal order.
        for v in sorted(adj[u]):
            edge = tuple(sorted((u, v)))
            if edge not in visited_edges:
                next_neighbor = v
                break

        if next_neighbor is not None:
            # If we found an unvisited neighbor, we go down that branch.
            v = next_neighbor
            visited_edges.add(tuple(sorted((u, v))))
            stack.append(v)
            path.append(v)
        else:
            # If there are no unvisited neighbors, we are at a dead end. Backtrack.
            stack.pop()
            if stack:
                # The new top of the stack is the parent, so we add it to the path
                # to represent the pen moving back.
                parent = stack[-1]
                path.append(parent)

    # The final backtrack might add the start node again. Let's clean it up.
    if len(path) > 1 and path[0] == path[-1]:
       return path[:-1]
       
    return path

# def get_node_labels_from_edge_labels(edge_index, pred_edge_labels, num_nodes):
#     """Computes node clusters from predicted edge labels via connected components."""
#     if isinstance(edge_index, torch.Tensor):
#         edge_index = edge_index.cpu().numpy()
#     positive_edges = edge_index[:, pred_edge_labels == 1]
#     pred_edges_undirected = {tuple(sorted(e)) for e in positive_edges.T}
#     if not pred_edges_undirected:
#         return np.arange(num_nodes)
#     row, col = zip(*pred_edges_undirected)
#     adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
#     n_components, node_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
#     return node_labels

def get_node_labels_from_edge_labels(edge_index, pred_edge_labels, num_nodes):
    """Computes node clusters from predicted edge labels via connected components."""
    logging.debug("=== get_node_labels_from_edge_labels called ===")
    logging.debug(f"Input num_nodes: {num_nodes}")
    logging.debug(f"edge_index type: {type(edge_index)}, shape: {edge_index.shape if hasattr(edge_index, 'shape') else 'N/A'}")
    logging.debug(f"pred_edge_labels type: {type(pred_edge_labels)}, shape: {pred_edge_labels.shape if hasattr(pred_edge_labels, 'shape') else 'N/A'}")
    
    # Convert tensors to numpy
    if isinstance(edge_index, torch.Tensor):
        logging.debug("Converting edge_index from torch.Tensor to numpy")
        edge_index = edge_index.cpu().numpy()
    if isinstance(pred_edge_labels, torch.Tensor):
        logging.debug("Converting pred_edge_labels from torch.Tensor to numpy")
        pred_edge_labels = pred_edge_labels.cpu().numpy()

    # Normalize shapes
    edge_index = np.atleast_2d(edge_index)
    logging.debug(f"After atleast_2d, edge_index shape: {edge_index.shape}")
    
    if edge_index.shape[0] != 2:
        logging.debug(f"Reshaping edge_index from {edge_index.shape} to (2, -1)")
        edge_index = edge_index.reshape(2, -1)
    
    pred_edge_labels = np.atleast_1d(pred_edge_labels)
    logging.debug(f"After atleast_1d, pred_edge_labels shape: {pred_edge_labels.shape}")

    # Handle trivial graph
    if edge_index.shape[1] == 0 or pred_edge_labels.size == 0:
        logging.info(f"Trivial graph detected: edge_index.shape[1]={edge_index.shape[1]}, "
                    f"pred_edge_labels.size={pred_edge_labels.size}. Returning isolated nodes.")
        return np.arange(num_nodes)

    # Select only positive edges
    mask = (pred_edge_labels == 1)
    logging.debug(f"Positive edge mask shape: {mask.shape}, sum: {np.sum(mask)}")
    
    if mask.ndim > 1:
        logging.debug(f"Flattening mask from shape {mask.shape}")
        mask = mask.flatten()
    
    positive_edges = edge_index[:, mask]
    logging.debug(f"positive_edges shape after masking: {positive_edges.shape}")

    # Handle case of no positive edges
    if positive_edges.size == 0:
        logging.info(f"No positive edges found. Returning {num_nodes} isolated nodes.")
        return np.arange(num_nodes)

    # Ensure shape is (2, N)
    if positive_edges.ndim == 1:
        logging.debug(f"Reshaping positive_edges from 1D (size={positive_edges.size}) to (2, 1)")
        positive_edges = positive_edges.reshape(2, 1)
    
    logging.debug(f"Final positive_edges shape: {positive_edges.shape} "
                 f"({positive_edges.shape[1]} edge(s))")

    # Convert to undirected edges - iterate by column index to avoid .T issues
    logging.debug("Building undirected edge set...")
    pred_edges_undirected = {
        tuple(sorted(positive_edges[:, i])) 
        for i in range(positive_edges.shape[1])
    }
    logging.debug(f"Created {len(pred_edges_undirected)} undirected edge(s)")
    
    if not pred_edges_undirected:
        logging.warning("pred_edges_undirected is empty after deduplication. Returning isolated nodes.")
        return np.arange(num_nodes)

    # Build adjacency and find connected components
    logging.debug("Building sparse adjacency matrix...")
    row, col = zip(*pred_edges_undirected)
    adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    
    logging.debug(f"Running connected_components on {num_nodes} nodes with {len(row)} edges...")
    n_components, node_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    
    logging.info(f"Found {n_components} connected component(s) for {num_nodes} nodes")
    logging.debug(f"Node label distribution: {np.bincount(node_labels)}")
    logging.debug("=== get_node_labels_from_edge_labels finished ===")
    
    return node_labels



def efficient_rand_index(true_labels, pred_labels):
    """Calculates the Rand Index efficiently using a contingency table."""
    contingency_table = pd.crosstab(true_labels, pred_labels)
    sum_ij = np.sum(contingency_table.values**2)
    sum_i = np.sum(np.sum(contingency_table, axis=1)**2)
    sum_j = np.sum(np.sum(contingency_table, axis=0)**2)
    n = len(true_labels)
    total_pairs = comb(n, 2)
    tp = 0.5 * (sum_ij - n)
    fp = 0.5 * (sum_j - sum_ij)
    fn = 0.5 * (sum_i - sum_ij)
    tn = total_pairs - (tp + fp + fn)
    return (tp + tn) / total_pairs if total_pairs > 0 else 1.0

def visualize_system_evaluation(
    pos, page_id, model_positive_edges, gt_mst_edges, input_edges_set, output_path: Path
):
    """
    Visualizes the full system evaluation, distinguishing between model errors and pipeline errors.
    - Green: True Positives (Correctly Kept)
    - Red: False Positives (Incorrectly Kept by Model)
    - Orange Dashed: Model False Negatives (Incorrectly Deleted by Model)
    - Purple Dotted: Structural False Negatives (Missed by Input Graph Pipeline)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tp_edges = model_positive_edges.intersection(gt_mst_edges)
    fp_edges = model_positive_edges - gt_mst_edges
    model_fn_edges = (input_edges_set.intersection(gt_mst_edges)) - model_positive_edges
    structural_fn_edges = gt_mst_edges - input_edges_set
    all_input_graph_edges = list(input_edges_set)

    plt.figure(figsize=(20, 16))
    plt.scatter(pos[:, 0], -pos[:, 1], s=15, c='black', zorder=5)
    plt.gca().set_aspect('equal', adjustable='box')

    def draw_edges(edges, color, linestyle='-', linewidth=1.0, zorder=1):
        if not edges: return
        for u, v in edges:
            plt.plot([pos[u, 0], pos[v, 0]], [-pos[u, 1], -pos[v, 1]], 
                     color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder)

    draw_edges(all_input_graph_edges, 'lightgrey', linewidth=0.7, zorder=1)
    draw_edges(fp_edges, 'red', linewidth=1.2, zorder=2)
    draw_edges(model_fn_edges, 'orange', linestyle='--', linewidth=1.8, zorder=3)
    draw_edges(structural_fn_edges, 'purple', linestyle=':', linewidth=2.0, zorder=4)
    draw_edges(tp_edges, 'green', linewidth=1.8, zorder=5)

    legend_elements = [
        Line2D([0], [0], color='lightgrey', lw=1, label='Candidate Edge'),
        Line2D([0], [0], color='green', lw=2, label='Correctly Kept (TP)'),
        Line2D([0], [0], color='red', lw=1.5, label='Incorrectly Kept (FP)'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Incorrectly Deleted by Model (Model FN)'),
        Line2D([0], [0], color='purple', lw=2, linestyle=':', label='Missed by Pipeline (Structural FN)'),
        plt.scatter([], [], s=30, color='black', label='Node (Character)')
    ]
    # plt.legend(handles=legend_elements, loc='best', fontsize='x-large')
    plt.title(f"System-Level Evaluation for Page: {page_id}", fontsize=20)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def create_page_xml(
    page_id,
    model_positive_edges,
    points_unnormalized,
    page_dims,
    output_path: Path,
    pred_node_labels: np.ndarray,
    polygons_data: dict,
    use_best_fit_line: bool = False,
    extend_percentage: float = 0.01
):
    """
    Generates a PAGE XML file. For each connected component, it creates a <TextLine>
    with a baseline and now also includes the <Coords> for the text line polygon.
    """
    PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ET.register_namespace('', PAGE_XML_NAMESPACE)

    num_nodes = len(points_unnormalized)

    adj = defaultdict(list)
    for u, v in model_positive_edges:
        adj[u].append(v)
        adj[v].append(u)

    components = find_connected_components(model_positive_edges, num_nodes)

    pc_gts = ET.Element(f"{{{PAGE_XML_NAMESPACE}}}PcGts")
    metadata = ET.SubElement(pc_gts, "Metadata")
    ET.SubElement(metadata, "Creator").text = "GNN-Prediction-Script"

    page = ET.SubElement(pc_gts, "Page", attrib={
        "imageFilename": f"{page_id}.jpg",
        "imageWidth": str(int(page_dims['width']*2)),
        "imageHeight": str(int(page_dims['height']*2))
    })

    min_x = np.min(points_unnormalized[:, 0])
    min_y = np.min(points_unnormalized[:, 1])
    max_x = np.max(points_unnormalized[:, 0])
    max_y = np.max(points_unnormalized[:, 1])
    region_coords = f"{int(min_x*2)},{int(min_y*2)} {int(max_x*2)},{int(min_y*2)} {int(max_x*2)},{int(max_y*2)} {int(min_x*2)},{int(max_y*2)}"

    text_region = ET.SubElement(page, "TextRegion", id="region_1")
    ET.SubElement(text_region, "Coords", points=region_coords)

    for component in components:
        if not component: continue
        
        # Get the points for the current component
        component_points = np.array([points_unnormalized[idx] for idx in component])
        
        if len(component_points) < 1:
            continue

        # Determine the label for this component to ensure consistent IDs
        line_label = pred_node_labels[component[0]]
            
        text_line = ET.SubElement(text_region, "TextLine", id=f"line_{line_label + 1}")
        
        if use_best_fit_line:
            baseline_points_for_fitting = np.array(
                [[p[0], p[1] + (p[2] / 2)] for p in component_points]
            )
            endpoints = fit_robust_line_and_extend(
                baseline_points_for_fitting, 
                extend_percentage=extend_percentage,
                robust_method='huber'
            )
            if endpoints:
                p1, p2 = endpoints
                baseline_points_str = f"{int(p1[0] * 2)},{int(p1[1] * 2)} {int(p2[0] * 2)},{int(p2[1] * 2)}"
            else:
                continue
        else:
            path_indices = trace_component_with_backtracking(component, adj)
            if len(path_indices) < 1: continue
            ordered_points = [points_unnormalized[idx] for idx in path_indices]
            baseline_points_str = " ".join([f"{int(p[0]*2)},{int((p[1]+(p[2]/2))*2)}" for p in ordered_points])

        ET.SubElement(text_line, "Baseline", points=baseline_points_str)
        
        # Add the corresponding polygon coordinates to the TextLine
        if line_label in polygons_data:
            polygon_points = polygons_data[line_label]
            coords_str = " ".join([f"{p[0]},{p[1]}" for p in polygon_points]) # we do not double the coords here, because we upscale the HEATMAP!
            ET.SubElement(text_line, "Coords", points=coords_str)
        else:
            logging.warning(f"Page {page_id}: No polygon data found for line label {line_label}, Coords tag will be omitted.")

    tree = ET.ElementTree(pc_gts)
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="\t", level=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)
# ===================================================================
#                       MAIN INFERENCE SCRIPT
# ===================================================================

def run_inference_with_eval(args):
    """Main function for system-level evaluation."""
    output_dir = Path(args.output_dir); setup_logging(output_dir / 'inference_with_eval.log'); device = get_device('auto')
    with open(args.dataset_config_path, 'r') as f: d_config = DatasetCreationConfig(**yaml.safe_load(f))
    gt_config = d_config.ground_truth
    checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
    model = checkpoint['model']; model.to(device); model.eval()
    model_name = Path(args.model_checkpoint).parent.name.split('-')[0]
    
    eval_output_dir = output_dir / f"{model_name}_eval_results_{pd.Timestamp.now():%Y%m%d_%H%M}"; eval_output_dir.mkdir(parents=True, exist_ok=True)
    # predictions_dataset_dir = Path(args.input_dir).parent / "gnn-dataset-pred"
    # predictions_dataset_dir.mkdir(exist_ok=True)
    # xml_output_dir = Path(args.input_dir).parent / "page-xml-graph-predictions" #fix syntax
    # xml_output_dir.mkdir(exist_ok=True)
    predictions_dataset_dir = eval_output_dir / "gnn-dataset-pred"
    predictions_dataset_dir.mkdir(exist_ok=True)
    xml_output_dir = eval_output_dir / "page-xml-graph-predictions" #fix syntax
    xml_output_dir.mkdir(exist_ok=True)



    
    if args.visualize:
        viz_dir = eval_output_dir / "visualizations"; viz_dir.mkdir(exist_ok=True)
    
    all_page_metrics = []
    input_files = sorted(list(Path(args.input_dir).glob('*_inputs_normalized.txt')))
    
    for file_path in input_files:
        page_id = file_path.name.replace('_inputs_normalized.txt', '')
        logging.info("--- Processing page: %s ---", page_id)


        # gt_labels_path = file_path.parent / f"{page_id}_labels_textline.txt"
        # if not gt_labels_path.exists(): raise FileNotFoundError(f"Required GT file not found: {gt_labels_path}")
        
        try:
            # gt_node_labels = np.loadtxt(gt_labels_path, dtype=int)
            points_normalized = np.loadtxt(file_path)
            if points_normalized.ndim == 1: points_normalized = points_normalized.reshape(1, -1)
            
            # --- START: TODO IMPLEMENTATION (Data Loading) ---
            # Load unnormalized points and dimensions for XML generation
            unnormalized_path = file_path.parent / f"{page_id}_inputs_unnormalized.txt"
            dims_path = file_path.parent / f"{page_id}_dims.txt"
            if not unnormalized_path.exists() or not dims_path.exists():
                logging.warning("Skipping XML generation for page %s: Unnormalized or dims file not found.", page_id)
                can_generate_xml = False
            else:
                points_unnormalized = np.loadtxt(unnormalized_path)
                if points_unnormalized.ndim == 1: points_unnormalized = points_unnormalized.reshape(1, -1)
                dims = np.loadtxt(dims_path)
                page_dims = {'width': dims[0], 'height': dims[1]}
                can_generate_xml = True
            # --- END: TODO IMPLEMENTATION (Data Loading) ---

            # if points_normalized.shape[0] != len(gt_node_labels):
            #      logging.warning("Skipping page %s: Mismatched counts. Points: %d, Labels: %d.", page_id, points_normalized.shape[0], len(gt_node_labels)); continue
        except Exception as e:
            logging.error("Could not load data for page %s: %s", page_id, e); continue

        # gt_mst_edges = create_ground_truth_graph_edges(points_normalized, gt_node_labels, gt_config)
        page_dims_norm = {'width': 1.0, 'height': 1.0} # Use normalized dims for graph creation
        input_graph_data = create_input_graph_edges(points_normalized, page_dims_norm, d_config.input_graph)
        input_edges_set = input_graph_data["edges"]

        if not input_edges_set:
            logging.warning("Skipping page %s: No candidate edges generated by input graph constructor.", page_id)
            # if gt_mst_edges: model_positive_edges = set()
            # else: continue
        else:
            edge_index_undirected = torch.tensor(list(input_edges_set), dtype=torch.long).t().contiguous()
            if d_config.input_graph.directionality == "bidirectional":
                edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
            else:
                edge_index = edge_index_undirected
            
            node_features = get_node_features(points_normalized, input_graph_data["heuristic_degrees"], d_config.features)
            edge_features = get_edge_features(edge_index, node_features, input_graph_data["heuristic_edge_counts"], d_config.features)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features).to(device)

            # with torch.no_grad():
            #     logits = model(data.x, data.edge_index, data.edge_attr)
            #     pred_edge_labels = torch.argmax(logits, dim=1).cpu().numpy()
            
            # model_positive_edges = {tuple(sorted(e)) for e in data.edge_index[:, pred_edge_labels == 1].cpu().numpy().T}
            threshold = 0.5  # adjust this for more precision

            with torch.no_grad():
                logits = model(data.x, data.edge_index, data.edge_attr)  # [num_edges, num_classes]
                probs = F.softmax(logits, dim=1)  # convert logits → probabilities
                
                # Take probability of class 1 (positive edge)
                pos_probs = probs[:, 1]
                
                # Apply threshold
                pred_edge_labels = (pos_probs > threshold).cpu().numpy().astype(int)

            model_positive_edges = {
                tuple(sorted(e)) 
                for e in data.edge_index[:, pred_edge_labels == 1].cpu().numpy().T
            }

        # For Rand Index and XML, we need the final node clusters from the model's predictions
        # This requires the edge_index tensor that was fed to the model
        pred_edge_index_tensor = torch.tensor(list(input_edges_set), dtype=torch.long).t()
        pred_node_labels_all_edges = torch.zeros(pred_edge_index_tensor.shape[1], dtype=torch.int32)
        # Create a boolean mask to identify positive predictions
        positive_edges_map = {edge: True for edge in model_positive_edges}
        for i in range(pred_edge_index_tensor.shape[1]):
            edge = tuple(sorted(pred_edge_index_tensor[:, i].tolist()))
            if edge in positive_edges_map:
                pred_node_labels_all_edges[i] = 1
                
        pred_node_labels = get_node_labels_from_edge_labels(pred_edge_index_tensor, pred_node_labels_all_edges, len(points_normalized))

        # --- START: FIX ---
        # Save the predicted labels and copy associated files to the prediction directory
        # This must be done BEFORE calling segmentLinesFromPointClusters.
        pred_labels_path = predictions_dataset_dir / f"{page_id}_labels_textline.txt"
        np.savetxt(pred_labels_path, pred_node_labels, fmt='%d')

        input_dir = file_path.parent
        for associated_file in input_dir.glob(f"{page_id}_*"):
            # This copies the unnormalized points and other necessary files.
            # It correctly skips the ground-truth labels file from the input directory.
            if associated_file.name == f"{page_id}_labels_textline.txt": continue
            shutil.copy(associated_file, predictions_dataset_dir / associated_file.name)
        # --- END: FIX ---

        # Generate line images and polygon data first, now that the files are in place.
        logging.info("Generating line images and polygon data for page %s...", page_id)
        polygons_data = segmentLinesFromPointClusters(Path(args.input_dir).parent, page_id, BINARIZE_THRESHOLD=args.BINARIZE_THRESHOLD, BBOX_PAD_V=args.BBOX_PAD_V, BBOX_PAD_H=args.BBOX_PAD_H, CC_SIZE_THRESHOLD_RATIO=args.CC_SIZE_THRESHOLD_RATIO, GNN_PRED_PATH=eval_output_dir)

        # Generate the PAGE XML, now including the polygon data
        if can_generate_xml:
            xml_path = xml_output_dir / f"{page_id}.xml"
            create_page_xml(
                page_id,
                model_positive_edges,
                points_unnormalized,
                page_dims,
                xml_path,
                pred_node_labels,   # Pass node labels for matching
                polygons_data,      # Pass the generated polygon data
                use_best_fit_line=False,
                extend_percentage=0.01
            )
            logging.info("Saved PAGE XML with polygon predictions to: %s", xml_path)




    #     # === START OF UNIFIED SYSTEM-LEVEL METRICS ===
    #     tp = len(model_positive_edges.intersection(gt_mst_edges))
    #     fp = len(model_positive_edges - gt_mst_edges)
    #     fn = len(gt_mst_edges - model_positive_edges)

    #     page_metrics = {'page_id': page_id}
    #     page_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0



    #     # --- START: NEW TEXTLINE F1-SCORE CALCULATION (OBJECT-LEVEL METRIC) ---
    #     num_nodes = len(points_normalized)

    #     # 1. Identify the ground-truth text lines (objects) from the GT edges
    #     if not gt_mst_edges:
    #         gt_edge_index = torch.empty((2, 0), dtype=torch.long)
    #     else:
    #         # Convert set of edge tuples to a tensor for the metric function
    #         gt_edge_index = torch.tensor(list(gt_mst_edges), dtype=torch.long).t()
    #     gt_edge_labels = torch.ones(gt_edge_index.shape[1], dtype=torch.long)
    #     gt_lines = get_textlines_from_edges(num_nodes, gt_edge_index, gt_edge_labels)

    #     # 2. Identify the predicted text lines (objects) from the model's positive edges
    #     if not model_positive_edges:
    #         pred_edge_index = torch.empty((2, 0), dtype=torch.long)
    #     else:
    #         pred_edge_index = torch.tensor(list(model_positive_edges), dtype=torch.long).t()
    #     pred_edge_labels = torch.ones(pred_edge_index.shape[1], dtype=torch.long)
    #     pred_lines = get_textlines_from_edges(num_nodes, pred_edge_index, pred_edge_labels)

    #     # 3. Compare the sets of objects to get TP, FP, FN counts for text lines
    #     tl_tp, tl_fp, tl_fn = calculate_textline_level_counts(gt_lines, pred_lines, iou_threshold=0.5)

    #     # 4. Calculate precision, recall, and F1 for this page's text lines
    #     tl_precision = tl_tp / (tl_tp + tl_fp) if (tl_tp + tl_fp) > 0 else 0.0
    #     tl_recall = tl_tp / (tl_tp + tl_fn) if (tl_tp + tl_fn) > 0 else 0.0
    #     tl_f1 = 2 * (tl_precision * tl_recall) / (tl_precision + tl_recall) if (tl_precision + tl_recall) > 0 else 0.0

    #     # 5. Add the new object-level metrics to the results dictionary
    #     page_metrics['textline_f1_score'] = tl_f1
    #     page_metrics['textline_precision'] = tl_precision
    #     page_metrics['textline_recall'] = tl_recall
    #     # --- END: NEW TEXTLINE F1-SCORE CALCULATION ---
    #     page_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    #     # This is the F1 score for the positive class, also called the binary F1 score.
    #     page_metrics['f1_score'] = 2 * (page_metrics['precision'] * page_metrics['recall']) / (page_metrics['precision'] + page_metrics['recall']) if (page_metrics['precision'] + page_metrics['recall']) > 0 else 0.0

    #     # --- START: MACRO F1 SCORE CALCULATION ---
    #     universe_edges = sorted(list(input_edges_set.union(gt_mst_edges)))
        
    #     if not universe_edges:
    #         page_metrics['macro_f1_score'] = 1.0 if not gt_mst_edges and not model_positive_edges else 0.0
    #     else:
    #         y_true = [1 if edge in gt_mst_edges else 0 for edge in universe_edges]
    #         y_pred = [1 if edge in model_positive_edges else 0 for edge in universe_edges]
    #         from sklearn.metrics import f1_score
    #         page_metrics['macro_f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    #     # --- END: MACRO F1 SCORE CALCULATION ---

    #     page_metrics['simplified_ged'] = fp + fn
    #     page_metrics['rand_index'] = efficient_rand_index(gt_node_labels, pred_node_labels)
        
    #     # The file writing/copying logic is now above, so it is removed from here.

    #     all_page_metrics.append(page_metrics)
    #     logging.info("System-Level Metrics for page %s: %s", page_id, {k: f'{v:.4f}' for k, v in page_metrics.items() if k != 'page_id'})
        
    #     if args.visualize:
    #         viz_path = viz_dir / f"{page_id}_system_evaluation.png"
    #         visualize_system_evaluation(points_normalized, page_id, model_positive_edges, gt_mst_edges, input_edges_set, viz_path)

    # if not all_page_metrics: logging.warning("No pages processed. No output CSV generated."); return
    # results_df = pd.DataFrame(all_page_metrics).set_index('page_id')
    # agg_metrics = results_df.agg(['mean', 'median']).round(4)
    # mode_metrics = results_df.mode(axis=0).iloc[0].rename('mode').round(4)
    # agg_metrics = pd.concat([agg_metrics, mode_metrics.to_frame().T])
    # logging.info("\n\n--- Aggregated System-Level Results ---\n%s", agg_metrics.to_string())
    # per_page_csv_path = eval_output_dir / 'per_page_system_metrics.csv'
    # aggregated_csv_path = eval_output_dir / 'aggregated_system_metrics.csv'
    # results_df.to_csv(per_page_csv_path); agg_metrics.to_csv(aggregated_csv_path)
    # logging.info("\nSaved per-page metrics to: %s\nSaved aggregated metrics to: %s\n", per_page_csv_path, aggregated_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GNN pipeline for a full system-level evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with _inputs_normalized.txt and _labels_textline.txt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results, visualizations, and logs.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained .pt model checkpoint.")
    parser.add_argument("--dataset_config_path", type=str, required=True, help="Path to the dataset creation config YAML. MUST match the one used for training.")
    parser.add_argument("--visualize", action="store_true", help="Generate and save system-level evaluation visualizations.")
    parser.add_argument("--BINARIZE_THRESHOLD", type=float, default=130, help="Threshold for binarizing the heatmap.")
    parser.add_argument("--BBOX_PAD_V", type=float, default=0.7, help="Vertical padding for bounding boxes in line segmentation.")
    parser.add_argument("--BBOX_PAD_H", type=float, default=0.5, help="Horizontal padding for bounding boxes in line segmentation.")
    parser.add_argument("--CC_SIZE_THRESHOLD_RATIO", type=float, default=0.4, help="Connected component size threshold ratio for line segmentation.")
    
    args = parser.parse_args()
    run_inference_with_eval(args)
    