# inference_with_eval.py


import torch
import numpy as np
import yaml
import logging
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch.nn.functional as F


import matplotlib
matplotlib.use('Agg')  # MUST be called before importing pyplot
import shutil
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from segment_from_point_clusters import segmentLinesFromPointClusters



from gnn_data_preparation.config_models import DatasetCreationConfig
from gnn_data_preparation.graph_constructor import create_input_graph_edges
from gnn_data_preparation.feature_engineering import get_node_features, get_edge_features
from gnn_data_preparation.utils import setup_logging
from torch_geometric.data import Data



def get_device(device_config: str) -> torch.device:
    """Gets the torch device based on config and availability, and logs the choice."""
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}") # This line confirms the choice in your logs
    return device



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
# ===================================================================
#                       MAIN INFERENCE SCRIPT
# ===================================================================

def run_gnn_inference(args):
    """Main function for system-level evaluation."""
    # 2. Set other arguments
    input_dir = f"{args.manuscript_path}/gnn-dataset"
    output_dir = f"{args.manuscript_path}/segmented_lines"

    setup_logging(Path(output_dir) / 'inference_with_eval.log')
    device = get_device('auto')

    with open(args.dataset_config_path, 'r') as f: 
        d_config = DatasetCreationConfig(**yaml.safe_load(f))
    checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
    model = checkpoint['model']
    model.to(device)
    model.eval()


    predictions_dataset_dir = Path(output_dir) / "gnn-format" # TODO use path to join
    predictions_dataset_dir.mkdir(exist_ok=True)
    xml_output_dir = Path(output_dir) / "page-xml-format" #fix syntax
    xml_output_dir.mkdir(exist_ok=True)


    
    input_files = sorted(list(Path(input_dir).glob('*_inputs_normalized.txt')))
    
    for file_path in input_files:
        page_id = file_path.name.replace('_inputs_normalized.txt', '')
        logging.info("--- Processing page: %s ---", page_id)
        try:
            points_normalized = np.loadtxt(file_path)
            if points_normalized.ndim == 1: points_normalized = points_normalized.reshape(1, -1)
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

        except Exception as e:
            logging.error("Could not load data for page %s: %s", page_id, e); continue


        page_dims_norm = {'width': 1.0, 'height': 1.0} # Use normalized dims for graph creation
        input_graph_data = create_input_graph_edges(points_normalized, page_dims_norm, d_config.input_graph)
        input_edges_set = input_graph_data["edges"]

        if not input_edges_set:
            logging.warning("Skipping page %s: No candidate edges generated by input graph constructor.", page_id)

        else:
            edge_index_undirected = torch.tensor(list(input_edges_set), dtype=torch.long).t().contiguous()
            if d_config.input_graph.directionality == "bidirectional":
                edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
            else:
                edge_index = edge_index_undirected
            
            node_features = get_node_features(points_normalized, input_graph_data["heuristic_degrees"], d_config.features)
            edge_features = get_edge_features(edge_index, node_features, input_graph_data["heuristic_edge_counts"], d_config.features)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features).to(device)

            threshold = 0.5  # this can be adjusted to optimize precision/recall trade-off

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
    
        # convert binary edge classification predictions to node labels
        pred_node_labels = get_node_labels_from_edge_labels(pred_edge_index_tensor, pred_node_labels_all_edges, len(points_normalized))

        # Save predicted node labels, and input files
        pred_labels_path = predictions_dataset_dir / f"{page_id}_labels_textline.txt"
        np.savetxt(pred_labels_path, pred_node_labels, fmt='%d')
        gnn_input_dir = file_path.parent
        for associated_file in gnn_input_dir.glob(f"{page_id}_*"):
            if associated_file.name == f"{page_id}_labels_textline.txt": continue
            shutil.copy(associated_file, predictions_dataset_dir / associated_file.name)


        # Generate line images and polygon data first, now that the files are in place.
        logging.info("Generating line images and polygon data for page %s...", page_id)
        polygons_data = segmentLinesFromPointClusters(Path(input_dir).parent, page_id, BINARIZE_THRESHOLD=args.BINARIZE_THRESHOLD, BBOX_PAD_V=args.BBOX_PAD_V, BBOX_PAD_H=args.BBOX_PAD_H, CC_SIZE_THRESHOLD_RATIO=args.CC_SIZE_THRESHOLD_RATIO, GNN_PRED_PATH=output_dir)

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

        #copy files from images_resized to segmented_lines/images_resized
        resized_images_src = Path(args.manuscript_path) / "images_resized"
        resized_images_dst = Path(output_dir) / "images_resized"
        resized_images_dst.mkdir(exist_ok=True)
        for img_file in resized_images_src.glob("*.jpg"):
            shutil.copy(img_file, resized_images_dst / img_file.name)



