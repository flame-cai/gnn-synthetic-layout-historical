As an expert in software development and deep learning (Graph Neural Networks), please assist me with the following code:
This code is part of a larger system which performs layout analysis on manuscript images using a Graph Neural Network (GNN). The layout analysis problem is formulated in a graph based manner, where characters are treated as nodes and characters of the same text lines are connected with edges. Thus nodes containing the same textline have the same text line label. The user can also label nodes with textbox labels, marking nodes of each text box with the same integer label. Once labelled (using gnn + manual corrections), the system generates PAGE XML files containing text regions and text lines, along with visualizations. The system also saves textline images. I want you help in imporving how the textline imags are saved. Right now the text line images are saved in a manner which do not repect the textbox boundaries. I want you to modify the code so that textline images are saved in a manner which respects textbox boundaries. In other words, create a folder for each textbox, and save all textline images belonging to that textbox in that folder.

Please look for opportunities to simplify and make the code more efficient while implementing this change, while keep the overall functionality intact.
For example please don't save the lines from file segment_from_point_clusters.py, instead save the line where the page_xml files are created. However ensure that this is functionally eqivalent, but just organized better as the purpose of segment_from_point_clusters.py is to get text line polygons from graph based node labels (where each node belonging to the same textline has the same label) not to save images. PLease think carefully about how to organize this. THE FUNCTIONALITY MUST REMAIN THE SAME, JUST ORGANIZED BETTER.

Please think carefully understand the flow of the code and implement precise changes to achieve this. Do not change any other functionality. Only modify the code related to saving textline images to respect textbox boundaries. If you think you need extra information, please ask before proceeding.


Here is the code:
```python




import torch
import numpy as np
import yaml
import logging
import shutil
from pathlib import Path
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data
import cv2
from datetime import datetime

# gnn_inference.py
import os
from collections import defaultdict
from gnn_data_preparation.utils import setup_logging
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import xml.etree.ElementTree as ET

from segment_from_point_clusters import segmentLinesFromPointClusters
from gnn_data_preparation.config_models import DatasetCreationConfig
from gnn_data_preparation.graph_constructor import create_input_graph_edges
from gnn_data_preparation.feature_engineering import get_node_features, get_edge_features

# Global Cache
LOADED_MODEL = None
LOADED_CONFIG = None
DEVICE = None

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_once(model_checkpoint_path, config_path):
    global LOADED_MODEL, LOADED_CONFIG, DEVICE
    if LOADED_MODEL is None:
        DEVICE = get_device()
        print(f"Loading model from {model_checkpoint_path} on {DEVICE}...")
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE, weights_only=False)
        LOADED_MODEL = checkpoint['model']
        LOADED_MODEL.to(DEVICE)
        LOADED_MODEL.eval()
        
        with open(config_path, 'r') as f:
            LOADED_CONFIG = DatasetCreationConfig(**yaml.safe_load(f))
    return LOADED_MODEL, LOADED_CONFIG, DEVICE


def generate_xml_and_images_for_page(manuscript_path, page_id, node_labels, graph_edges, args_dict, textbox_labels=None, nodes=None):
    """
    Saves user corrections and regenerates XML.
    Handles coordinate scaling: Frontend (Image Space) -> Storage (Heatmap Space).
    """
    base_path = Path(manuscript_path)
    raw_input_dir = base_path / "gnn-dataset"
    output_dir = base_path / "layout_analysis_output"
    gnn_format_dir = output_dir / "gnn-format"
    gnn_format_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Heatmap Dimensions (Crucial for scaling)
    raw_dims_path = raw_input_dir / f"{page_id}_dims.txt"
    if not raw_dims_path.exists():
        # Fallback for rare edge cases
        raw_dims_path = gnn_format_dir / f"{page_id}_dims.txt"
        
    dims = np.loadtxt(raw_dims_path) 
    heatmap_w, heatmap_h = dims[0], dims[1]
    max_dim_heatmap = max(heatmap_w, heatmap_h)

    points_unnormalized = []
    points_normalized = []

    if nodes is not None:
        # --- SCALING FIX ---
        # Frontend sends Image Space coordinates. Storage expects Heatmap Space.
        # Image Space is 2x Heatmap Space.
        scale_factor = 0.5 

        for n in nodes:
            # 1. Image Space
            img_x, img_y, img_s = float(n['x']), float(n['y']), float(n['s'])
            
            # 2. Heatmap Space (Storage)
            hm_x, hm_y, hm_s = img_x * scale_factor, img_y * scale_factor, img_s * scale_factor
            
            points_unnormalized.append([hm_x, hm_y, hm_s])
            
            # 3. Normalized (0-1) for GNN
            norm_x, norm_y, norm_s = hm_x / max_dim_heatmap, hm_y / max_dim_heatmap, hm_s / max_dim_heatmap
            points_normalized.append([norm_x, norm_y, norm_s])
            
        points_unnormalized = np.array(points_unnormalized)
        points_normalized = np.array(points_normalized)
        
        # Save NEW node definitions
        np.savetxt(gnn_format_dir / f"{page_id}_inputs_unnormalized.txt", points_unnormalized, fmt='%f')
        np.savetxt(gnn_format_dir / f"{page_id}_inputs_normalized.txt", points_normalized, fmt='%f')
        # Always copy dims to history so it is self-contained
        if raw_dims_path.exists():
            shutil.copy(raw_dims_path, gnn_format_dir / f"{page_id}_dims.txt")
        
    else:
        # --- HISTORY PROTECTION ---
        # If frontend didn't send nodes (legacy call), rely on what's on disk.
        # Check if we already have modified files; if not, copy from raw.
        if not (gnn_format_dir / f"{page_id}_inputs_unnormalized.txt").exists():
            for suffix in ["_inputs_normalized.txt", "_inputs_unnormalized.txt", "_dims.txt"]:
                src = raw_input_dir / f"{page_id}{suffix}"
                dst = gnn_format_dir / f"{page_id}{suffix}"
                if src.exists(): shutil.copy(src, dst)
        
        points_unnormalized = np.loadtxt(gnn_format_dir / f"{page_id}_inputs_unnormalized.txt")
        if points_unnormalized.size == 0:
            points_unnormalized = np.empty((0, 3))
        elif points_unnormalized.ndim == 1: 
            points_unnormalized = points_unnormalized.reshape(1, -1)


    # 2. Save Corrected Edges
    unique_edges = set()
    num_nodes = len(points_unnormalized)
    
    for e in graph_edges:
        if 'source' in e and 'target' in e:
            u, v = sorted((int(e['source']), int(e['target'])))
            if u < num_nodes and v < num_nodes:
                unique_edges.add((u, v))
            
    edges_save_path = gnn_format_dir / f"{page_id}_edges.txt"
    if unique_edges:
        np.savetxt(edges_save_path, list(unique_edges), fmt='%d')
    else:
        open(edges_save_path, 'w').close()

    # 3. Calculate Structural Labels
    if unique_edges:
        row, col = zip(*unique_edges)
        data = np.ones(len(row) + len(col))
        adj = csr_matrix((data, (list(row)+list(col), list(col)+list(row))), shape=(num_nodes, num_nodes))
    else:
        adj = csr_matrix((num_nodes, num_nodes))

    n_components, final_structural_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    np.savetxt(gnn_format_dir / f"{page_id}_labels_textline.txt", final_structural_labels, fmt='%d')

    # 4. Save Textbox Labels
    final_textbox_labels = np.zeros(num_nodes, dtype=int)
    if textbox_labels is not None:
        if len(textbox_labels) == num_nodes:
            final_textbox_labels = np.array(textbox_labels, dtype=int)
            np.savetxt(gnn_format_dir / f"{page_id}_labels_textbox.txt", final_textbox_labels, fmt='%d')
        else:
             print(f"Warning: Textbox label count {len(textbox_labels)} != Node count {num_nodes}. Resetting.")
             
    # 5. Run Segmentation
    polygons_data = segmentLinesFromPointClusters(
        str(output_dir.parent), 
        page_id, 
        BINARIZE_THRESHOLD=args_dict.get('BINARIZE_THRESHOLD', 0.5098), 
        BBOX_PAD_V=args_dict.get('BBOX_PAD_V', 0.7), 
        BBOX_PAD_H=args_dict.get('BBOX_PAD_H', 0.5), 
        CC_SIZE_THRESHOLD_RATIO=args_dict.get('CC_SIZE_THRESHOLD_RATIO', 0.4), 
        GNN_PRED_PATH=str(output_dir)
    )

    xml_output_dir = output_dir / "page-xml-format"
    xml_output_dir.mkdir(exist_ok=True)
    
    # 6. Generate XML
    create_page_xml(
        page_id,
        unique_edges,
        points_unnormalized,
        {'width': heatmap_w, 'height': heatmap_h}, 
        xml_output_dir / f"{page_id}.xml",
        final_structural_labels, 
        polygons_data,
        textbox_labels=final_textbox_labels,
        image_path=base_path / "images_resized" / f"{page_id}.jpg",
    )

    resized_images_dst_dir = output_dir / "images_resized"
    resized_images_dst_dir.mkdir(exist_ok=True)
    src_img = base_path / "images_resized" / f"{page_id}.jpg"
    if src_img.exists():
        shutil.copy(src_img, resized_images_dst_dir / f"{page_id}.jpg")

    return {"status": "success", "lines": len(polygons_data)}




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

def run_gnn_prediction_for_page(manuscript_path, page_id, model_path, config_path):
    print(f"Fetching data for page: {page_id}")
    
    base_path = Path(manuscript_path)
    raw_input_dir = base_path / "gnn-dataset"               
    history_dir = base_path / "layout_analysis_output" / "gnn-format" 
    
    # --- 1. Load Node Data (Prioritize Modified History) ---
    modified_norm_path = history_dir / f"{page_id}_inputs_normalized.txt"
    modified_dims_path = history_dir / f"{page_id}_dims.txt"
    
    if modified_norm_path.exists() and modified_dims_path.exists():
        print(f"--> Loading USER-MODIFIED node definitions from {history_dir}")
        file_path = modified_norm_path
        dims_path = modified_dims_path
    else:
        print(f"--> Loading RAW CRAFT node definitions from {raw_input_dir}")
        file_path = raw_input_dir / f"{page_id}_inputs_normalized.txt"
        dims_path = raw_input_dir / f"{page_id}_dims.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Data for page {page_id} not found.")

    # Handle empty files (if user deleted all nodes previously)
    try:
        points_normalized = np.loadtxt(file_path)
    except UserWarning:
        points_normalized = np.array([])

    if points_normalized.size == 0:
        points_normalized = np.empty((0, 3))
    elif points_normalized.ndim == 1: 
        points_normalized = points_normalized.reshape(1, -1)
    
    dims = np.loadtxt(dims_path)
    full_width = dims[0] * 2
    full_height = dims[1] * 2
    max_dimension = max(full_width, full_height)
    
    nodes_payload = [
        {
            "x": float(p[0]) * max_dimension, 
            "y": float(p[1]) * max_dimension, 
            "s": float(p[2])
        } 
        for p in points_normalized
    ]
    
    response = {
        "nodes": nodes_payload,
        "edges": [],
        "textline_labels": [-1] * len(points_normalized),
        "textbox_labels": [],
        "dimensions": [full_width, full_height]
    }

    # --- 2. Check for Saved Topology (Edges/Labels) ---
    saved_edges_path = history_dir / f"{page_id}_edges.txt"
    saved_labels_path = history_dir / f"{page_id}_labels_textline.txt"
    saved_textbox_path = history_dir / f"{page_id}_labels_textbox.txt"
    
    if saved_edges_path.exists():
        print(f"Found saved edge topology...")
        saved_edges = []
        try:
            if saved_edges_path.stat().st_size > 0:
                raw_edges = np.loadtxt(saved_edges_path, dtype=int, ndmin=2)
                if raw_edges.ndim == 1 and raw_edges.size >= 2:
                    raw_edges = raw_edges.reshape(1, -1)
                
                for row in raw_edges:
                    if len(row) >= 2:
                        saved_edges.append({
                            "source": int(row[0]),
                            "target": int(row[1]),
                            "label": 1
                        })
        except Exception as e:
            print(f"Warning reading edges: {e}")
            
        response["edges"] = saved_edges
        
        if saved_labels_path.exists():
            try:
                labels = np.loadtxt(saved_labels_path, dtype=int)
                if labels.size == len(points_normalized):
                     response["textline_labels"] = labels.tolist()
            except Exception: pass 
        
        if saved_textbox_path.exists():
            try:
                tb_labels = np.loadtxt(saved_textbox_path, dtype=int)
                if tb_labels.size == len(points_normalized):
                    response["textbox_labels"] = tb_labels.tolist()
            except Exception: pass

        return response

    # --- 3. Run GNN (Only if no history exists) ---
    if len(points_normalized) == 0:
        return response

    print(f"Running GNN Inference...")
    model, d_config, device = load_model_once(model_path, config_path)
    
    page_dims_norm = {'width': 1.0, 'height': 1.0}
    input_graph_data = create_input_graph_edges(points_normalized, page_dims_norm, d_config.input_graph)
    input_edges_set = input_graph_data["edges"]

    if not input_edges_set:
        return response

    edge_index_undirected = torch.tensor(list(input_edges_set), dtype=torch.long).t().contiguous()
    if d_config.input_graph.directionality == "bidirectional":
        edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
    else:
        edge_index = edge_index_undirected

    node_features = get_node_features(points_normalized, input_graph_data["heuristic_degrees"], d_config.features)
    edge_features = get_edge_features(edge_index, node_features, input_graph_data["heuristic_edge_counts"], d_config.features)
    
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features).to(device)

    threshold = 0.5
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits, dim=1)
        pred_edge_labels = (probs[:, 1] > threshold).cpu().numpy()

    model_positive_edges = set()
    edge_index_cpu = data.edge_index.cpu().numpy()
    
    for idx, is_pos in enumerate(pred_edge_labels):
        if is_pos:
            u, v = edge_index_cpu[:, idx]
            model_positive_edges.add(tuple(sorted((u, v))))

    final_edges = []
    for u, v in input_edges_set:
        if tuple(sorted((u, v))) in model_positive_edges:
            final_edges.append({"source": int(u), "target": int(v), "label": 1})

    response["edges"] = final_edges
    return response



def create_page_xml(
    page_id,
    model_positive_edges,
    points_unnormalized,
    page_dims,
    output_path: Path,
    pred_node_labels: np.ndarray,
    polygons_data: dict,
    textbox_labels: np.ndarray = None,
    use_best_fit_line: bool = False,
    extend_percentage: float = 0.01,
    image_path: Path = None, 
    save_vis: bool = True
):
    """
    Generates a PAGE XML file with reading order and textregions (textboxes).
    """
    PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ET.register_namespace('', PAGE_XML_NAMESPACE)

    num_nodes = len(points_unnormalized)

    # Build Adjacency List
    adj = defaultdict(list)
    for u, v in model_positive_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Find Connected Components (Text Lines)
    components = find_connected_components(model_positive_edges, num_nodes)
    
    # -- Data Structure Preparation --
    # Group components (lines) by their Textbox Label
    # Map: textbox_id -> list of component indices
    regions = defaultdict(list)
    
    for i, component in enumerate(components):
        if not component: continue
        
        # Determine Textbox Label for this line
        # We take the majority label of nodes in the component
        comp_tb_labels = []
        if textbox_labels is not None:
             for node_idx in component:
                 comp_tb_labels.append(textbox_labels[node_idx])
        
        if comp_tb_labels:
            # majority vote
            tb_id = np.bincount(comp_tb_labels).argmax()
        else:
            tb_id = 0 # Default region
            
        regions[tb_id].append(component)

    # -- PAGE XML Setup --
    pc_gts = ET.Element(f"{{{PAGE_XML_NAMESPACE}}}PcGts")
    metadata = ET.SubElement(pc_gts, "Metadata")
    ET.SubElement(metadata, "Creator").text = "GNN-Layout-Analysis"
    ET.SubElement(metadata, "Created").text = datetime.now().isoformat()
    

    final_w = int(page_dims['width'] * 2)
    final_h = int(page_dims['height'] * 2)

    page = ET.SubElement(pc_gts, "Page", attrib={
        "imageFilename": f"{page_id}.jpg",
        "imageWidth": str(final_w),
        "imageHeight": str(final_h)
    })

    # -- Visualization Setup --
    vis_img = None
    if save_vis:
        if image_path and image_path.exists():
            vis_img = cv2.imread(str(image_path))
            if vis_img is not None and (vis_img.shape[0] != final_h or vis_img.shape[1] != final_w):
                vis_img = cv2.resize(vis_img, (final_w, final_h))
        if vis_img is None:
            vis_img = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    # -- Sorting Logic Helper --
    def get_centroid(comp_nodes):
        # Scale coords by 2 to match XML space
        xs = [points_unnormalized[n][0] * 2 for n in comp_nodes]
        ys = [points_unnormalized[n][1] * 2 for n in comp_nodes]
        return np.mean(xs), np.mean(ys)

    # 1. Sort Regions (Textboxes)
    # We sort regions based on the centroid of all lines within them.
    # Reading order: Top-to-Bottom, then Left-to-Right.
    region_centroids = []
    for tb_id, comps in regions.items():
        all_nodes = [n for comp in comps for n in comp]
        if not all_nodes: continue
        cx, cy = get_centroid(all_nodes)
        region_centroids.append({'id': tb_id, 'cx': cx, 'cy': cy})
    
    # Sort primarily by Y, secondarily by X (with some tolerance for Y lines)
    # A simple approach: Y + (X * small_factor) usually works for mostly vertical layouts, 
    # but for standard reading order we often want strict Top-Down.
    # Let's use strict Y for region sorting for now.
    region_centroids.sort(key=lambda r: (r['cy'], r['cx']))

# -- Construct XML Hierarchy --
    for r_idx, region_info in enumerate(region_centroids):
        tb_id = region_info['id']
        comps = regions[tb_id]
        
        # --- FIXED AREA CALCULATION ---
        # Instead of using node centers, we collect all coordinate points 
        # from the polygons of the lines assigned to this region.
        region_xs = []
        region_ys = []
        
        for comp in comps:
            # Get the line label (cluster ID) to retrieve the specific polygon
            line_label = pred_node_labels[comp[0]]
            
            if line_label in polygons_data and len(polygons_data[line_label]) > 0:
                # Use the polygon points (assuming they are already in the target 2x scale 
                # consistent with their usage later in the script)
                poly_pts = polygons_data[line_label]
                for p in poly_pts:
                    region_xs.append(p[0])
                    region_ys.append(p[1])
            else:
                # Fallback: If no polygon exists, use the node centers (scaled by 2)
                # This prevents a crash if segmentation data is missing for a line
                for n in comp:
                    region_xs.append(points_unnormalized[n][0] * 2)
                    region_ys.append(points_unnormalized[n][1] * 2)
        
        if not region_xs: 
            continue # Skip empty regions

        min_x, max_x = min(region_xs), max(region_xs)
        min_y, max_y = min(region_ys), max(region_ys)
        # ------------------------------
        
        region_elem = ET.SubElement(page, "TextRegion", id=f"region_{r_idx}", custom=f"textbox_label_{tb_id}")
        region_coords_str = f"{int(min_x)},{int(min_y)} {int(max_x)},{int(min_y)} {int(max_x)},{int(max_y)} {int(min_x)},{int(max_y)}"
        ET.SubElement(region_elem, "Coords", points=region_coords_str)

        # Visualize Region (Yellow)
        if save_vis and vis_img is not None:
            cv2.rectangle(vis_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 255), 2)
            cv2.putText(vis_img, f"R{r_idx}", (int(min_x), int(min_y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 2. Sort Lines within Region
        # Standard reading order: Top-to-Bottom
        comp_centroids = []
        for comp in comps:
            cx, cy = get_centroid(comp)
            comp_centroids.append({'comp': comp, 'cx': cx, 'cy': cy})
        
        comp_centroids.sort(key=lambda c: c['cy'])

        for l_idx, line_info in enumerate(comp_centroids):
            component = line_info['comp']
            line_label = pred_node_labels[component[0]] # Original cluster ID
            
            text_line = ET.SubElement(region_elem, "TextLine", id=f"region_{r_idx}_line_{l_idx}")
            
            # --- Baseline Calculation ---
            baseline_points_str = ""
            baseline_vis = []
            
            path_indices = trace_component_with_backtracking(component, adj)
            if len(path_indices) >= 1:
                ordered_points = [points_unnormalized[idx] for idx in path_indices]
                # Scale by 2 and shift Y to center of char (y + s/2)
                baseline_vis = [[int(p[0]*2), int((p[1]+(p[2]/2))*2)] for p in ordered_points]
                baseline_points_str = " ".join([f"{p[0]},{p[1]}" for p in baseline_vis])
            
            ET.SubElement(text_line, "Baseline", points=baseline_points_str)

            # --- Polygon Coords ---
            polygon_vis = []
            if line_label in polygons_data:
                polygon_points = polygons_data[line_label] # already scaled in segment script? NO, usually 1x
                # segmentLinesFromPointClusters output is usually based on 'image' size.
                # Since points_unnormalized are 1x, and XML is 2x, check segment script.
                # segment script resizes heatmaps up. Assuming polygons match final_w/final_h space.
                # If polygons_data are from 1x coords, we need to scale. 
                # segmentLinesFromPointClusters uses upscale_heatmap=True -> matches original image size (2x of heatmap dims)
                # So polygon_points should be in 2x scale already.
                coords_str = " ".join([f"{p[0]},{p[1]}" for p in polygon_points])
                ET.SubElement(text_line, "Coords", points=coords_str)
                polygon_vis = polygon_points

            # Visualize Line
            if save_vis and vis_img is not None:
                if len(polygon_vis) > 0:
                    pts = np.array(polygon_vis, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
                if len(baseline_vis) > 0:
                    pts = np.array(baseline_vis, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts], False, (0, 0, 255), 2)

    # Save XML
    tree = ET.ElementTree(pc_gts)
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="\t", level=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)

    # Save Visualization
    if save_vis and vis_img is not None:
        vis_output_path = output_path.parent / f"{output_path.stem}_viz.jpg"
        cv2.imwrite(str(vis_output_path), vis_img)


















# segement_from_point_clusters.py

import os
import shutil
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import UnivariateSpline
import math
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend (fast)
import matplotlib.pyplot as plt
from skimage import io


def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def resize_with_padding(image, target_size, background_color=(0, 0, 0)):
    """
    Resizes an image to a target size while maintaining its aspect ratio by padding.
    """
    target_w, target_h = target_size
    if image is None or image.size == 0:
        return np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
        
    h, w = image.shape[:2]

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded_image = np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    
    return padded_image

def visualize_projection_profile(profile, crop_shape, orientation='horizontal', color=(255, 255, 255), thickness=1):
    """
    Visualizes a 1D projection profile, creating an image that corresponds to the
    dimensions of the original crop.
    """
    if profile is None or len(profile) == 0:
        return np.zeros((crop_shape[0], crop_shape[1], 3), dtype=np.uint8)

    crop_h, crop_w = crop_shape
    max_val = np.max(profile)
    if max_val == 0:
        max_val = 1

    if orientation == 'horizontal':
        vis_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        for i, val in enumerate(profile):
            length = int((val / max_val) * crop_w)
            if i < crop_h:
                cv2.line(vis_image, (0, i), (length, i), color, thickness)
    else:  # vertical
        vis_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        for i, val in enumerate(profile):
            length = int((val / max_val) * crop_h)
            if i < crop_w:
                cv2.line(vis_image, (i, crop_h - 1), (i, crop_h - 1 - length), color, thickness)
            
    return vis_image

def create_debug_collage(original_uncropped, padded_crop, cleaned_blob, component_viz_img, heatmap_crop, config):
    """
    Creates a 2x4 collage of debugging images for a single bounding box.
    """
    TILE_SIZE = (200, 200)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4
    FONT_COLOR = (255, 255, 255)

    # Prepare heatmap-related visualizations
    _, bin_heat_crop = cv2.threshold(heatmap_crop, config['BINARIZE_THRESHOLD'], 255, cv2.THRESH_BINARY)
    h_prof_heat = np.sum(bin_heat_crop, axis=1) / 255
    v_prof_heat = np.sum(bin_heat_crop, axis=0) / 255
    v_prof_heat_viz = visualize_projection_profile(v_prof_heat, bin_heat_crop.shape, 'vertical', color=(0, 0, 255))
    h_prof_heat_viz = visualize_projection_profile(h_prof_heat, bin_heat_crop.shape, 'horizontal', color=(0, 0, 255))
    heatmap_colorized = cv2.applyColorMap(heatmap_crop, cv2.COLORMAP_JET)

    # Create resized tiles for the collage
    orig_uncropped_tile = resize_with_padding(original_uncropped, TILE_SIZE)
    padded_crop_tile = resize_with_padding(padded_crop, TILE_SIZE)
    cleaned_blob_tile = resize_with_padding(cleaned_blob, TILE_SIZE)
    component_viz_tile = resize_with_padding(component_viz_img, TILE_SIZE)
    heat_crop_tile = resize_with_padding(heatmap_colorized, TILE_SIZE)
    v_prof_heat_tile = resize_with_padding(v_prof_heat_viz, TILE_SIZE)
    h_prof_heat_tile = resize_with_padding(h_prof_heat_viz, TILE_SIZE)
    bin_heat_tile = resize_with_padding(bin_heat_crop, TILE_SIZE)

    # Add labels to each tile
    cv2.putText(orig_uncropped_tile, "Original BBox", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(padded_crop_tile, "Padded BBox", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(cleaned_blob_tile, "Cleaned Blob", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(component_viz_tile, "Analyzed Components", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(heat_crop_tile, "Heatmap", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(v_prof_heat_tile, "V-Profile (Heat)", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(h_prof_heat_tile, "H-Profile (Heat)", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(bin_heat_tile, "Binarized Heatmap", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Assemble the collage
    row1 = cv2.hconcat([orig_uncropped_tile, padded_crop_tile, cleaned_blob_tile, component_viz_tile])
    row2 = cv2.hconcat([heat_crop_tile, v_prof_heat_tile, h_prof_heat_tile, bin_heat_tile])
    collage = cv2.vconcat([row1, row2])
    
    return collage


def analyze_and_clean_blob(blob, line_type, config):
    """
    Analyzes connected components in a blob, identifies noise touching boundaries,
    and returns a cropped version of the blob along with crop coordinates.
    """
    if blob.size == 0:
        return blob, np.zeros_like(blob, dtype=np.uint8), [0, 0, 0, 0]

    _, bin_blob = cv2.threshold(blob, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_blob, connectivity=8)
    
    component_viz_img = np.zeros((blob.shape[0], blob.shape[1], 3), dtype=np.uint8)
    component_viz_img[labels != 0] = [255, 0, 0] # Default: Blue for valid components

    crop_h, crop_w = blob.shape
    crop_coords = [0, crop_h, 0, crop_w]

    if num_labels > 1:
        for i in range(1, num_labels):
            x_c, y_c, w_c, h_c, _ = stats[i]
            
            is_touching_boundary = (y_c == 0 or y_c + h_c == crop_h) if line_type != 'vertical' else (x_c == 0 or x_c + w_c == crop_w)
            is_size_constrained = (h_c <= config['CC_SIZE_THRESHOLD_RATIO'] * crop_h) if line_type != 'vertical' else (w_c <= config['CC_SIZE_THRESHOLD_RATIO'] * crop_w)

            if is_touching_boundary and is_size_constrained:
                component_viz_img[labels == i] = [0, 0, 255] # Red for noise
                if line_type != 'vertical':
                    if y_c == 0: crop_coords[0] = max(crop_coords[0], y_c + h_c)
                    if y_c + h_c == crop_h: crop_coords[1] = min(crop_coords[1], y_c)
                else:
                    if x_c == 0: crop_coords[2] = max(crop_coords[2], x_c + w_c)
                    if x_c + w_c == crop_w: crop_coords[3] = min(crop_coords[3], x_c)
    
    if crop_coords[0] >= crop_coords[1] or crop_coords[2] >= crop_coords[3]:
        return np.array([]), np.array([]), [0, 0, 0, 0]

    top, bottom, left, right = crop_coords
    cleaned_blob = blob[top:bottom, left:right]
    final_viz_img = component_viz_img[top:bottom, left:right]

    return cleaned_blob, final_viz_img, crop_coords


def gen_bounding_boxes(det, binarize_threshold):
    """
    Generates bounding boxes from a 2D heatmap loaded from an image file.

    This function assumes the input `det` is a NumPy array representing an image
    with pixel values in the [0, 255] range. It converts a normalized
    threshold (0.0 to 1.0) to this scale and then finds contours.

    Args:
        det (np.ndarray): The 2D input heatmap, assumed to be on a [0, 255] scale.
        binarize_threshold (float): The normalized threshold to apply.
                                    This value should be between 0.0 and 1.0.

    Returns:
        list[tuple]: A list of bounding boxes in the format (x, y, w, h).
    """
    # 1. Ensure the input is in the correct data type for OpenCV functions.
    #    We do NOT re-normalize the value range, as it's assumed to be 0-255.
    img = np.uint8(det)
    threshold_val = int(binarize_threshold * 255)
    _, img1 = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]




# def load_node_features_and_labels(points_file, labels_file):
#     points = np.loadtxt(points_file, dtype=float, ndmin=2).astype(int)
#     with open(labels_file, "r") as f: labels = [line.strip() for line in f]
#     features, filtered_labels = [], []
#     for point, label in zip(points, labels):
#         if label.lower() != "none":
#             features.append(point)
#             filtered_labels.append(int(label))
#     return np.array(features), np.array(filtered_labels)
def load_node_features_and_labels(points_file, labels_file):
    # RED TEAM FIX: Handle empty files gracefully
    try:
        points = np.loadtxt(points_file, dtype=float, ndmin=2)
        if points.size == 0:
            return np.array([]), np.array([])
        
        with open(labels_file, "r") as f: labels = [line.strip() for line in f]
        
        features, filtered_labels = [], []
        # Handle case where labels file might be empty or mismatched
        if len(labels) != len(points):
             return np.array([]), np.array([])

        for point, label in zip(points, labels):
            if label.lower() != "none":
                features.append(point)
                filtered_labels.append(int(label))
        return np.array(features), np.array(filtered_labels)
    except Exception as e:
        print(f"Warning loading features/labels: {e}")
        return np.array([]), np.array([])



def assign_labels_and_plot(bounding_boxes, points, labels, image, output_path):
    labeled_bboxes = []
    
    # Track which point indices have been assigned to a box to handle manually added nodes
    matched_point_indices = set()
    
    # 1. Standard assignment: Box -> Points
    for x_min, y_min, w, h in bounding_boxes:
        x_max, y_max = x_min + w, y_min + h
        
        # Identify points strictly inside this box
        # We store (point_data, label, original_index)
        pts_in_box = []
        for i, (p, lab) in enumerate(zip(points, labels)):
            if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max:
                pts_in_box.append((p, lab, i))
        
        if not pts_in_box:
            continue
            
        # Check if all points inside share the same label
        unique_labels = {item[1] for item in pts_in_box}
        
        if len(unique_labels) == 1:
            # Simple case: One label for the whole box
            seg_label = pts_in_box[0][1]
            labeled_bboxes.append((x_min, y_min, w, h, seg_label))
            
            # Mark these points as matched
            for _, _, idx in pts_in_box:
                matched_point_indices.add(idx)
                
        else:
            # Complex case: Multiple labels in one box (e.g., lines touching vertically)
            # Sort points by Y coordinate
            pts_in_box.sort(key=lambda item: item[0][1])
            
            # Define split boundaries based on midpoints between label changes
            boundaries = [y_min]
            for i in range(1, len(pts_in_box)):
                curr_lab = pts_in_box[i][1]
                prev_lab = pts_in_box[i-1][1]
                if curr_lab != prev_lab:
                    mid_y = int((pts_in_box[i][0][1] + pts_in_box[i-1][0][1]) / 2)
                    boundaries.append(max(y_min, min(y_max, mid_y)))
            boundaries.append(y_max)
            
            # Create sub-boxes
            for i in range(1, len(boundaries)):
                top, bot = boundaries[i-1], boundaries[i]
                if bot <= top: continue
                
                # Determine label for this segment (use the first point falling in range)
                seg_label = next((lab for p, lab, idx in pts_in_box if top <= p[1] <= bot), None)
                
                if seg_label is not None:
                    labeled_bboxes.append((x_min, top, w, bot - top, seg_label))
                    
                    # Mark points in this segment as matched
                    for p, lab, idx in pts_in_box:
                        if top <= p[1] <= bot:
                            matched_point_indices.add(idx)

    # 2. Hacky Fix for Added Nodes: Create artificial boxes for unmatched points
    # Calculate median box size to use as default for 'ghost' boxes
    median_w = 40 # Reasonable default
    median_h = 40
    if bounding_boxes:
        widths = [b[2] for b in bounding_boxes]
        heights = [b[3] for b in bounding_boxes]
        if widths: median_w = np.median(widths)
        if heights: median_h = np.median(heights)
        
    img_h, img_w = image.shape[:2] if image is not None else (10000, 10000)

    for i, (point, label) in enumerate(zip(points, labels)):
        if i not in matched_point_indices:
            # This point wasn't inside any CRAFT box (e.g., manual addition)
            px, py = int(point[0]), int(point[1])
            
            # Use median dimensions centered on the point
            w = int(median_w)
            h = int(median_h)
            
            x = px - w // 2
            y = py - h // 2
            
            # Clip to image boundaries
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            
            # Ensure w, h fit within image
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            if w > 0 and h > 0:
                labeled_bboxes.append((x, y, w, h, label))

    return labeled_bboxes

def detect_line_type(boxes):
    if len(boxes) < 2: return 'horizontal', None
    centers = sorted([(x + w//2, y + h//2) for x, y, w, h, _ in boxes], key=lambda p: p[0])
    x_coords, y_coords = [p[0] for p in centers], [p[1] for p in centers]
    x_range, y_range = (max(coords) - min(coords) for coords in (x_coords, y_coords)) if centers else (0, 0)
    if x_range < y_range * 0.3: return 'vertical', None
    if y_range < x_range * 0.3: return 'horizontal', None
    try:
        X, y = np.array(x_coords).reshape(-1, 1), np.array(y_coords)
        ransac = RANSACRegressor(random_state=42).fit(X, y)
        if ransac.score(X, y) > 0.85: return 'slanted', {'slope': ransac.estimator_.coef_[0], 'intercept': ransac.estimator_.intercept_}
        return 'curved', {'spline': UnivariateSpline(x_coords, y_coords, s=len(centers)*2)}
    except: return 'horizontal', None

def transform_boxes_to_horizontal(boxes, line_type, params):
    if line_type == 'horizontal': return boxes
    t_boxes = []
    if line_type == 'vertical':
        for x, y, w, h, label in boxes: t_boxes.append((y, -x - w, h, w, label))
    elif line_type == 'slanted' and params:
        angle = math.atan(params['slope'])
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        for x, y, w, h, label in boxes:
            cx, cy = x + w//2, y + h//2
            t_boxes.append((int(cx*cos_a - cy*sin_a - w/2), int(cx*sin_a + cy*cos_a - h/2), w, h, label))
    else: return boxes
    return t_boxes

def normalize_coordinates(boxes):
    if not boxes: return []
    min_x, min_y = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [(x - min_x, y - min_y, w, h, label) for x, y, w, h, label in boxes]

def crop_img(img):
    mask = img != int(np.median(img))
    if not np.any(mask): return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def get_bboxes_for_lines(img, unique_labels, bounding_boxes, debug_mode=False, debug_info=None):
    """
    Generates cropped line images. Uses DYNAMIC PADDING for the initial cleaning crop
    and STATIC PADDING from the config for the final canvas border.
    """
    line_images_data = []
    line_bounding_boxes_data = {label: [] for label in unique_labels}
    box_counter = 0
    config = debug_info.get('CONFIG', {}) if debug_info else {}


    for l in unique_labels:
        filtered_boxes = [box for box in bounding_boxes if box[4] == l]
        if not filtered_boxes:
            continue

        line_type, params = detect_line_type(filtered_boxes)

        if line_type == 'horizontal':
            PADDING_RATIO_V = config.get('BBOX_PAD_V', 0.7) # Default to 70% if not in config
            PADDING_RATIO_H = config.get('BBOX_PAD_H', 0.5) # Default to 50% if not in config
        else:
            PADDING_RATIO_V = config.get('BBOX_PAD_H', 0.5) # Default to 50% if not in config
            PADDING_RATIO_H = config.get('BBOX_PAD_V', 0.7) # Default to 70% if not in config

        cleaned_blobs_for_line = []
        final_coords_for_line = []

        for box in filtered_boxes:
            box_counter += 1
            orig_x, orig_y, orig_w, orig_h, _ = box
            try:
                # Get the original, unpadded crop for debugging.
                original_uncropped_blob = img[orig_y:orig_y + orig_h, orig_x:orig_x + orig_w]
                
                # Use dynamic, ratio-based padding for the initial crop to help cleaning.
                dynamic_pad_v = int(orig_h * PADDING_RATIO_V)
                dynamic_pad_h = int(orig_w * PADDING_RATIO_H)

                y1 = max(0, orig_y - dynamic_pad_v)
                y2 = orig_y + orig_h + dynamic_pad_v
                x1 = max(0, orig_x - dynamic_pad_h)
                x2 = orig_x + orig_w + dynamic_pad_h

                blob = img[y1:y2, x1:x2]
                if blob.size == 0:
                    continue

                cleaned_blob, component_viz_img, crop_coords = analyze_and_clean_blob(blob, line_type, config)
                
                if cleaned_blob.size == 0:
                    if debug_mode and debug_info:
                        # Even if the blob is empty, save a debug collage to see why
                        det_resized = debug_info.get('det_resized')
                        if det_resized is not None:
                            heatmap_crop = det_resized[y1:y2, x1:x2]
                            collage = create_debug_collage(original_uncropped_blob, blob, cleaned_blob, 
                                                           component_viz_img, heatmap_crop, config)
                            cv2.imwrite(os.path.join(debug_info['DEBUG_DIR'], f"line_{l:03d}_box_{box_counter:04d}_EMPTY.jpg"), collage)
                    continue

                c_top, _, c_left, _ = crop_coords
                final_box_x = x1 + c_left
                final_box_y = y1 + c_top
                final_box_w = cleaned_blob.shape[1]
                final_box_h = cleaned_blob.shape[0]

                if final_box_w > 0 and final_box_h > 0:
                    cleaned_blobs_for_line.append(cleaned_blob)
                    final_coords_for_line.append([final_box_x, final_box_y, final_box_w, final_box_h])

                if debug_mode and debug_info:
                    det_resized = debug_info.get('det_resized')
                    if det_resized is not None:
                        heatmap_crop = det_resized[y1:y2, x1:x2]
                        if heatmap_crop.size > 0:
                            collage = create_debug_collage(original_uncropped_blob, blob, cleaned_blob,
                                                           component_viz_img, heatmap_crop, config)
                            cv2.imwrite(os.path.join(debug_info['DEBUG_DIR'], f"line_{l:03d}_box_{box_counter:04d}.jpg"), collage)

            except Exception as e:
                print(f"Warning: Skipped box during analysis in line {l}: {e}")
        
        line_bounding_boxes_data[l] = final_coords_for_line
        
        
    return line_bounding_boxes_data


def segmentLinesFromPointClusters(BASE_PATH, page, upscale_heatmap=True, debug_mode=False, BINARIZE_THRESHOLD=0.30, BBOX_PAD_V=0.7, BBOX_PAD_H=0.5, CC_SIZE_THRESHOLD_RATIO=0.4, GNN_PRED_PATH=''):
    IMAGE_FILEPATH = os.path.join(BASE_PATH, "images_resized", f"{page}.jpg")
    HEATMAP_FILEPATH = os.path.join(BASE_PATH, "heatmaps", f"{page}.jpg")
    POINTS_FILEPATH = os.path.join(GNN_PRED_PATH, "gnn-format", f"{page}_inputs_unnormalized.txt")
    LABELS_FILEPATH = os.path.join(GNN_PRED_PATH, "gnn-format", f"{page}_labels_textline.txt")
    LINES_DIR = os.path.join(GNN_PRED_PATH, "image-format", page)
    DEBUG_DIR = os.path.join(GNN_PRED_PATH, "debug", page)
    POLY_VISUALIZATIONS_DIR = os.path.join(DEBUG_DIR, "poly_visualizations")

    # The polygon directory is no longer needed as polygons are now saved in the XML
    if os.path.exists(LINES_DIR): shutil.rmtree(LINES_DIR)
    os.makedirs(LINES_DIR)

    image = loadImage(IMAGE_FILEPATH)
    det = loadImage(HEATMAP_FILEPATH)
    if det.ndim == 3: det = det[:, :, 0]

    h_img, w_img = image.shape[:2]; h_heat, w_heat = det.shape[:2]
    features, labels = load_node_features_and_labels(POINTS_FILEPATH, LABELS_FILEPATH)

    if upscale_heatmap:
        det_resized = cv2.resize(det, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        processing_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if features.size > 0:
            features[:, :2] = (features[:, :2].astype(np.float64) * [w_img / w_heat, h_img / h_heat]).astype(int)
    else:
        image = cv2.resize(image, (w_heat, h_heat))
        h_img, w_img = image.shape[:2]
        det_resized = det
        processing_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    CONFIG = {
        'BINARIZE_THRESHOLD': BINARIZE_THRESHOLD,
        'CC_SIZE_THRESHOLD_RATIO': CC_SIZE_THRESHOLD_RATIO,
        'PAGE_MEDIAN_COLOR': int(np.median(processing_image))
        # Padding ratios for initial cleaning crop
        ,'BBOX_PAD_V': BBOX_PAD_V # 70% vertical padding for horizontal lines
        ,'BBOX_PAD_H': BBOX_PAD_H # 50% horizontal
    }

    bounding_boxes = gen_bounding_boxes(det_resized, CONFIG['BINARIZE_THRESHOLD'])
    labeled_bboxes = assign_labels_and_plot(bounding_boxes, features, labels, image.copy(),
                                            output_path=os.path.join(BASE_PATH, "heatmaps", f"{page}_all_labelled_boxes.jpg"))

    unique_labels = sorted(list(set(b[4] for b in labeled_bboxes)))

    # debug_info = None
    debug_info = {"DEBUG_DIR": DEBUG_DIR, "det_resized": det_resized, "CONFIG": CONFIG}
    if upscale_heatmap and debug_mode:
        print("Debug mode is ON.")
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        os.makedirs(POLY_VISUALIZATIONS_DIR)
        # debug_info = {"DEBUG_DIR": DEBUG_DIR, "det_resized": det_resized, "CONFIG": CONFIG}

    line_bounding_boxes_data = get_bboxes_for_lines(processing_image, unique_labels, labeled_bboxes,
                                                    debug_mode=(upscale_heatmap and debug_mode), debug_info=debug_info)

    poly_viz_page_img = image.copy()
    colors = [plt.cm.get_cmap('hsv', len(unique_labels) + 1)(i) for i in range(len(unique_labels))]
    label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}
    line_polygons_data = {}  # To store polygon data for returning

    for line_label, cleaned_boxes in line_bounding_boxes_data.items():
        if not cleaned_boxes: continue

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for (x, y, w, h) in cleaned_boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            # save the mask for debugging if needed
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 1:
            avg_line_height = np.mean([box[3] for box in cleaned_boxes])
            box_groups = [[] for _ in contours]
            for box in cleaned_boxes:
                center_x, center_y = box[0] + box[2] // 2, box[1] + box[3] // 2
                for i, contour in enumerate(contours):
                    point_to_test = (float(center_x), float(center_y))
                    if cv2.pointPolygonTest(contour, point_to_test, False) >= 0:
                        box_groups[i].append(box)
                        break
            
            box_groups = [group for group in box_groups if group]
            if len(box_groups) <= 1:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                connected_groups = [box_groups[0]]
                unconnected_groups = box_groups[1:]

                while unconnected_groups:
                    best_dist = float('inf')
                    closest_box_pair = (None, None)
                    group_to_add_index = -1

                    for i, u_group in enumerate(unconnected_groups):
                        for c_group in connected_groups:
                            for u_box in u_group:
                                for c_box in c_group:
                                    dist = np.linalg.norm(
                                        np.array([u_box[0] + u_box[2]/2, u_box[1] + u_box[3]/2]) -
                                        np.array([c_box[0] + c_box[2]/2, c_box[1] + c_box[3]/2])
                                    )
                                    if dist < best_dist:
                                        best_dist = dist
                                        closest_box_pair = (u_box, c_box)
                                        group_to_add_index = i
                    
                    if closest_box_pair[0] is not None:
                        box1, box2 = closest_box_pair
                        left_box, right_box = (box1, box2) if box1[0] < box2[0] else (box2, box1)
                        
                        y_center1 = left_box[1] + left_box[3] / 2
                        y_center2 = right_box[1] + right_box[3] / 2
                        bridge_y_center = (y_center1 + y_center2) / 2
                        
                        bridge_y1 = int(bridge_y_center - avg_line_height / 2)
                        bridge_y2 = int(bridge_y_center + avg_line_height / 2)
                        bridge_x1 = left_box[0] + left_box[2]
                        bridge_x2 = right_box[0]
                        
                        cv2.rectangle(mask, (bridge_x1, bridge_y1), (bridge_x2, bridge_y2), 255, -1)

                    connected_groups.append(unconnected_groups.pop(group_to_add_index))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            polygon = max(contours, key=cv2.contourArea)
            line_filename_base = f"line{line_label+1:03d}"
            
            # Instead of saving to JSON, store the polygon points in the dictionary
            polygon_points_xy = [point[0].tolist() for point in polygon]
            line_polygons_data[line_label] = polygon_points_xy

            if upscale_heatmap and debug_mode:
                color_idx = label_to_color_idx.get(line_label, 0)
                color = tuple(c * 255 for c in colors[color_idx][:3])
                cv2.drawContours(poly_viz_page_img, [polygon], -1, color, 2)
            
            # Save the cropped polygon area as the line image (this functionality remains)
            x, y, w, h = cv2.boundingRect(polygon)
            cropped_line_image = processing_image[y:y+h, x:x+w]
            new_img = np.ones(cropped_line_image.shape, dtype=np.uint8) * CONFIG['PAGE_MEDIAN_COLOR']
            mask_polygon = np.zeros(cropped_line_image.shape[:2], dtype=np.uint8)
            polygon_shifted = polygon - [x, y]
            cv2.drawContours(mask_polygon, [polygon_shifted], -1, 255, -1)
            new_img[mask_polygon == 255] = cropped_line_image[mask_polygon == 255]
            cv2.imwrite(os.path.join(LINES_DIR, f"{line_filename_base}.jpg"), new_img)

    if upscale_heatmap and debug_mode:
        viz_path = os.path.join(POLY_VISUALIZATIONS_DIR, f"{page}_all_polygons.jpg")
        cv2.imwrite(viz_path, poly_viz_page_img)
        print(f"Polygon visualization saved to {viz_path}")

    print(f"Successfully generated {len(line_polygons_data)} line images and polygon data.")
    return line_polygons_data