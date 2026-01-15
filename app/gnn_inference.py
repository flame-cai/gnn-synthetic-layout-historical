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