# gnn_inference.py
import torch
import numpy as np
import yaml
import logging
import shutil
from pathlib import Path
import torch.nn.functional as F
import os
from collections import defaultdict
from segment_from_point_clusters import segmentLinesFromPointClusters
from gnn_data_preparation.config_models import DatasetCreationConfig
from gnn_data_preparation.graph_constructor import create_input_graph_edges
from gnn_data_preparation.feature_engineering import get_node_features, get_edge_features
from gnn_data_preparation.utils import setup_logging
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import xml.etree.ElementTree as ET



# Global model cache to avoid reloading on every request
LOADED_MODEL = None
LOADED_CONFIG = None
DEVICE = None

def get_device(device_config='auto'):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_once(model_checkpoint_path, config_path):
    global LOADED_MODEL, LOADED_CONFIG, DEVICE
    if LOADED_MODEL is None:
        DEVICE = get_device()
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE, weights_only=False)
        LOADED_MODEL = checkpoint['model']
        LOADED_MODEL.to(DEVICE)
        LOADED_MODEL.eval()
        
        with open(config_path, 'r') as f:
            LOADED_CONFIG = DatasetCreationConfig(**yaml.safe_load(f))
    return LOADED_MODEL, LOADED_CONFIG, DEVICE

def get_node_labels_from_edge_labels(edge_index, pred_edge_labels, num_nodes):

    
    if isinstance(edge_index, torch.Tensor): edge_index = edge_index.cpu().numpy()
    if isinstance(pred_edge_labels, torch.Tensor): pred_edge_labels = pred_edge_labels.cpu().numpy()

    edge_index = np.atleast_2d(edge_index)
    if edge_index.shape[0] != 2: edge_index = edge_index.reshape(2, -1)
    
    mask = (pred_edge_labels == 1)
    positive_edges = edge_index[:, mask]
    
    if positive_edges.size == 0: return np.arange(num_nodes)

    pred_edges_undirected = {tuple(sorted(positive_edges[:, i])) for i in range(positive_edges.shape[1])}
    if not pred_edges_undirected: return np.arange(num_nodes)

    row, col = zip(*pred_edges_undirected)
    adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    n_components, node_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    return node_labels

# --- STAGE 1: Inference Only ---
def run_gnn_prediction_for_page(manuscript_path, page_id, model_path, config_path):
    """
    Runs Steps 1-4: Loads data, constructs graph, runs GNN.
    Returns the graph data (Nodes + Edges) to the API. 
    We do NOT calculate connected components here; the frontend visualizes edges directly.
    """
    print("Running GNN prediction for page:", page_id)
    model, d_config, device = load_model_once(model_path, config_path)
    
    input_dir = Path(manuscript_path) / "gnn-dataset"
    file_path = input_dir / f"{page_id}_inputs_normalized.txt"
    dims_path = input_dir / f"{page_id}_dims.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data for page {page_id} not found.")

    points_normalized = np.loadtxt(file_path)
    if points_normalized.ndim == 1: points_normalized = points_normalized.reshape(1, -1)
    
    # Load dimensions
    dims = np.loadtxt(dims_path)
    # The image dimensions sent to frontend need to be doubled (width*2, height*2) 
    # because the original heatmap processing (step 2) downsizes by 2.
    full_width = dims[0] * 2
    full_height = dims[1] * 2
    max_dimension = max(full_width, full_height)
    
    # Construct Graph
    page_dims_norm = {'width': 1.0, 'height': 1.0}
    input_graph_data = create_input_graph_edges(points_normalized, page_dims_norm, d_config.input_graph)
    input_edges_set = input_graph_data["edges"]

    # Initialize empty response structure
    # We initialize region_labels with -1 (Unlabeled)
    # The frontend will simply show nodes as grey until the user interacts or edits edges.
    initial_region_labels = [-1] * len(points_normalized)

    # If no edges, return trivial result
    if not input_edges_set:
        return {
            "nodes": [
                {"x": float(p[0]) * max_dimension, "y": float(p[1]) * max_dimension, "s": float(p[2])} 
                for p in points_normalized
            ],
            "edges": [],
            "region_labels": initial_region_labels,
            "dimensions": [full_width, full_height]
        }

    # Prepare PyG Data
    edge_index_undirected = torch.tensor(list(input_edges_set), dtype=torch.long).t().contiguous()
    if d_config.input_graph.directionality == "bidirectional":
        edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
    else:
        edge_index = edge_index_undirected

    node_features = get_node_features(points_normalized, input_graph_data["heuristic_degrees"], d_config.features)
    edge_features = get_edge_features(edge_index, node_features, input_graph_data["heuristic_edge_counts"], d_config.features)
    
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features).to(device)

    # Inference
    threshold = 0.5
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits, dim=1)
        pos_probs = probs[:, 1]
        pred_edge_labels = (pos_probs > threshold).cpu().numpy().astype(int)

    # Reconstruct positive edges
    model_positive_edges = set()
    edge_index_cpu = data.edge_index.cpu().numpy()
    for idx, is_pos in enumerate(pred_edge_labels):
        if is_pos:
            u, v = edge_index_cpu[:, idx]
            model_positive_edges.add(tuple(sorted((u, v))))

    positive_edges_list = []
    for u, v in input_edges_set:
        if tuple(sorted((u, v))) in model_positive_edges:
            positive_edges_list.append({
                "source": int(u), 
                "target": int(v), 
                "label": 1
            })

    # Format for Frontend
    graph_response = {
        "nodes": [
            {
                # Denormalize coordinates for pixels
                "x": float(p[0]) * max_dimension, 
                "y": float(p[1]) * max_dimension, 
                "s": float(p[2])
            } 
            for p in points_normalized
        ],
        "edges": positive_edges_list, # <--- ONLY SEND POSITIVE EDGES
        "region_labels": initial_region_labels,
        "dimensions": [full_width, full_height]
    }
    
    return graph_response
    
# --- STAGE 2: Generation ---
def generate_xml_and_images_for_page(manuscript_path, page_id, node_labels, graph_edges, args_dict):
    """
    Runs Step 5: Uses corrected graph to generate XML and line images.
    
    CHANGE: This now enforces STRUCTURAL LABELING. 
    It ignores the 'node_labels' array (semantic regions) and calculates 
    Line IDs purely based on the Connected Components of the 'graph_edges'.
    """
    input_dir = Path(manuscript_path) / "gnn-dataset"
    output_dir = Path(manuscript_path) / "segmented_lines"
    gnn_format_dir = output_dir / "gnn-format"
    gnn_format_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Reconstruct Adjacency Matrix from Frontend Edges
    # We filter for label==1, assuming the frontend sends us the valid positive edges.
    model_positive_edges = {tuple(sorted((e['source'], e['target']))) for e in graph_edges if e.get('label') == 1}
    
    # We need the number of nodes. We can infer it from the length of node_labels list 
    # (even though we ignore the values, the length is correct).
    num_nodes = len(node_labels)
    
    # 2. Calculate Structural Labels (Connected Components)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    if model_positive_edges:
        row, col = zip(*model_positive_edges)
        # Create symmetric adjacency for undirected graph
        all_rows = list(row) + list(col)
        all_cols = list(col) + list(row)
        data = np.ones(len(all_rows))
        adj = csr_matrix((data, (all_rows, all_cols)), shape=(num_nodes, num_nodes))
    else:
        # Handle case with no edges (all isolated nodes)
        adj = csr_matrix((num_nodes, num_nodes))

    # This generates IDs: 0, 1, 2... for every distinct interconnected group (Line)
    n_components, final_structural_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        
    # 3. Save these Structural Labels
    # This overwrites the file, ensuring the next steps use the corrected structure.
    np.savetxt(gnn_format_dir / f"{page_id}_labels_textline.txt", final_structural_labels, fmt='%d')
    
    # Copy other required files
    for suffix in ["_inputs_normalized.txt", "_inputs_unnormalized.txt", "_dims.txt"]:
        src = input_dir / f"{page_id}{suffix}"
        if src.exists():
            shutil.copy(src, gnn_format_dir / src.name)

    # Load unnormalized points for XML
    unnorm_path = input_dir / f"{page_id}_inputs_unnormalized.txt"
    points_unnormalized = np.loadtxt(unnorm_path)
    if points_unnormalized.ndim == 1: points_unnormalized = points_unnormalized.reshape(1, -1)
    
    dims_path = input_dir / f"{page_id}_dims.txt"
    dims = np.loadtxt(dims_path)
    page_dims = {'width': dims[0], 'height': dims[1]}

    # Segmentation (uses the labels_textline.txt we just saved)
    polygons_data = segmentLinesFromPointClusters(
        str(Path(input_dir).parent), 
        page_id, 
        BINARIZE_THRESHOLD=args_dict.get('BINARIZE_THRESHOLD', 0.5098), 
        BBOX_PAD_V=args_dict.get('BBOX_PAD_V', 0.7), 
        BBOX_PAD_H=args_dict.get('BBOX_PAD_H', 0.5), 
        CC_SIZE_THRESHOLD_RATIO=args_dict.get('CC_SIZE_THRESHOLD_RATIO', 0.4), 
        GNN_PRED_PATH=str(output_dir)
    )

    # Generate XML
    xml_output_dir = output_dir / "page-xml-format"
    xml_output_dir.mkdir(exist_ok=True)
    
    from gnn_inference import create_page_xml 
    
    create_page_xml(
        page_id,
        model_positive_edges,
        points_unnormalized,
        page_dims,
        xml_output_dir / f"{page_id}.xml",
        final_structural_labels, # Use the calculated structural labels
        polygons_data
    )

    # Copy resized image
    resized_images_src = Path(manuscript_path) / "images_resized"
    resized_images_dst = output_dir / "images_resized"
    resized_images_dst.mkdir(exist_ok=True)
    img_src = resized_images_src / f"{page_id}.jpg"
    if img_src.exists():
        shutil.copy(img_src, resized_images_dst / img_src.name)

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
# ==================================================================

