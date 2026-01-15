# data_creation/main_create_dataset.py
import argparse
import yaml
import logging
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config_models import DatasetCreationConfig
from graph_constructor import create_input_graph_edges, create_ground_truth_graph_edges
from feature_engineering import get_node_features, get_edge_features
from dataset_generator import HistoricalLayoutGNNDataset, create_sklearn_dataframe
from utils import setup_logging, find_page_ids
from torch_geometric.data import Data

def process_page(page_id: str, data_dir: Path, config: DatasetCreationConfig):
    """Processes a single page from a given directory to create a PyG Data object."""
    try:
        # 1. Load data from .txt files using the provided data_dir
        dims_path = data_dir / f"{page_id}_dims.txt"
        inputs_norm_path = data_dir / f"{page_id}_inputs_normalized.txt"
        labels_path = data_dir / f"{page_id}_labels_textline.txt"

        if not all([dims_path.exists(), inputs_norm_path.exists(), labels_path.exists()]):
            logging.warning(f"Skipping page {page_id}: missing one or more required files in {data_dir}.")
            return None

        if inputs_norm_path.stat().st_size == 0 or labels_path.stat().st_size == 0:
            logging.warning(f"Skipping page {page_id}: input or label file is empty.")
            return None

        page_dims_arr = np.loadtxt(dims_path)
        page_dims = {'width': page_dims_arr[0], 'height': page_dims_arr[1]}
        points_normalized = np.loadtxt(inputs_norm_path)
        
        if points_normalized.ndim == 1:
            points_normalized = points_normalized.reshape(1, -1)
            
        textline_labels = np.loadtxt(labels_path, dtype=int)

        if len(points_normalized) < config.min_nodes_per_page:
            logging.info(f"Skipping page {page_id}: has {len(points_normalized)} nodes, less than min {config.min_nodes_per_page}.")
            return None
            
    except Exception as e:
        logging.error(f"Error loading or processing initial data for page {page_id}: {e}")
        return None

    # 2. Construct Input and Ground Truth Graphs (Unchanged)
    input_graph_data = create_input_graph_edges(points_normalized, page_dims, config.input_graph)
    gt_edges_set = create_ground_truth_graph_edges(points_normalized, textline_labels, config.ground_truth)

    # 3. Create edge_index and edge labels (y) (Unchanged)
    input_edges = list(input_graph_data["edges"])
    if not input_edges:
        logging.warning(f"Skipping page {page_id}: no edges were generated for the input graph.")
        return None
        
    edge_index_undirected = torch.tensor(input_edges, dtype=torch.long).t().contiguous()
    edge_y = torch.tensor([1 if tuple(e) in gt_edges_set else 0 for e in input_edges], dtype=torch.long)
    
    if config.input_graph.directionality == "bidirectional":
        edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
        edge_y = torch.cat([edge_y, edge_y], dim=0)
    else: # unidirectional
        edge_index = edge_index_undirected

    # 4. Feature Engineering (Unchanged)
    node_features = get_node_features(points_normalized, input_graph_data["heuristic_degrees"], config.features)
    edge_features = get_edge_features(edge_index, node_features, input_graph_data["heuristic_edge_counts"], config.features)
    
    # 5. Assemble PyG Data object (Unchanged)
    data = Data(x=node_features, edge_index=edge_index, edge_y=edge_y)
    if edge_features is not None:
        data.edge_attr = edge_features
        
    if config.features.use_page_aspect_ratio:
        data.page_aspect_ratio = torch.tensor([page_dims['width'] / page_dims['height'] if page_dims['height'] > 0 else 1.0])

    data.page_id = page_id
    data.num_nodes = len(points_normalized)
    
    # Sanity checks (Unchanged)
    assert data.x.shape[0] == data.num_nodes, "Node feature dimension mismatch"
    assert data.edge_index.shape[1] == data.edge_y.shape[0], "Edge index and edge label mismatch"
    if data.edge_attr is not None:
        assert data.edge_index.shape[1] == data.edge_attr.shape[0], "Edge index and edge attribute mismatch"
        
    return data

def main():
    # --- NEW: Setup argument parsing ---
    parser = argparse.ArgumentParser(description="Create a dataset with predefined training and validation/test splits.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/1_dataset_creation_config.yaml',
        help='Path to the YAML configuration file.'
    )
    parser.add_argument('--output_dir', type=str, help='Override the main output directory.')
    parser.add_argument('--train_data_dir', type=str, help='Override the input directory for training data.')
    parser.add_argument('--val_test_data_dir', type=str, help='Override the input directory for validation/test data.')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    assert config_path.exists(), f"Configuration file not found at {config_path}"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = DatasetCreationConfig(**config_dict)

    # --- MODIFIED: Prioritize command-line arguments for directories ---
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir)
    train_data_dir = Path(args.train_data_dir) if args.train_data_dir else Path(config.train_data_dir)
    val_test_data_dir = Path(args.val_test_data_dir) if args.val_test_data_dir else Path(config.val_test_data_dir)

    # Setup output directory and logging
    dataset_version_dir = output_dir #/ f"{config.manuscript_name}-v{config.version}"
    dataset_version_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(dataset_version_dir / "dataset_creation.log")
    
    logging.info("Starting dataset creation process with predefined splits...")
    logging.info(f"Configuration (with potential overrides):\n"
                 f"  output_dir: {output_dir}\n"
                 f"  train_data_dir: {train_data_dir}\n"
                 f"  val_test_data_dir: {val_test_data_dir}")

    # Validate input directories
    logging.info(f"Training data source: {train_data_dir}")
    logging.info(f"Validation/Test data source: {val_test_data_dir}")
    assert train_data_dir.is_dir(), f"Training data directory not found: {train_data_dir}"
    assert val_test_data_dir.is_dir(), f"Validation/Test data directory not found: {val_test_data_dir}"

    # Find page IDs from respective directories
    train_page_ids = find_page_ids(train_data_dir)
    val_test_page_ids = find_page_ids(val_test_data_dir)
    
    logging.info(f"Found {len(train_page_ids)} pages for the training set.")
    logging.info(f"Found {len(val_test_page_ids)} pages to be split into validation and test sets.")
    
    if not train_page_ids or not val_test_page_ids:
        logging.critical("One of the source directories is empty. Cannot proceed.")
        return

    # Split the validation/test data
    val_ratio = config.splitting.val_ratio
    logging.info(f"Splitting validation/test data: {val_ratio*100}% validation, {(1-val_ratio)*100:.0f}% test.")
    
    rng = np.random.default_rng(config.splitting.random_seed)
    val_test_ids_arr = np.array(val_test_page_ids)
    rng.shuffle(val_test_ids_arr)
    
    split_index = int(len(val_test_ids_arr) * val_ratio)
    val_ids = val_test_ids_arr[:split_index]
    test_ids = val_test_ids_arr[split_index:]
    
    # Define the final splits
    splits = {
        "train": train_page_ids,
        "val": list(val_ids),
        "test": list(test_ids),
    }

    logging.info(f"Final split sizes: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    page_id_to_dir_map = {page_id: train_data_dir for page_id in splits['train']}
    page_id_to_dir_map.update({page_id: val_test_data_dir for page_id in splits['val']})
    page_id_to_dir_map.update({page_id: val_test_data_dir for page_id in splits['test']})

    # To maintain a similar output structure, we simulate a single "fold"
    fold_idx = 0 
    fold_dir = dataset_version_dir / "folds" / f"fold_{fold_idx}"
    logging.info(f"===== Processing all splits into '{fold_dir}' =====")

    for split_name, split_page_ids in splits.items():
        if not split_page_ids:
            logging.warning(f"Skipping {split_name} split as it contains no page IDs.")
            continue
            
        logging.info(f"--- Processing {split_name} split with {len(split_page_ids)} pages ---")
        
        data_list = []
        page_map = {} 

        for page_id in tqdm(split_page_ids, desc=f"Processing {split_name}"):
            source_dir = page_id_to_dir_map[page_id]
            graph_data = process_page(page_id, source_dir, config)
            if graph_data:
                page_map[len(data_list)] = page_id
                data_list.append(graph_data)

        if not data_list:
            logging.warning(f"No data generated for {split_name} split. Skipping save steps.")
            continue

        # Save GNN Dataset
        gnn_dir = fold_dir / "gnn" / split_name
        gnn_dir.mkdir(parents=True, exist_ok=True)
        HistoricalLayoutGNNDataset(root=str(gnn_dir), data_list=data_list)
        logging.info(f"Saved GNN {split_name} dataset to '{gnn_dir}'")
        
        # Save Sklearn Dataset
        if config.sklearn_format.enabled:
            sklearn_dir = fold_dir / "sklearn"
            sklearn_dir.mkdir(parents=True, exist_ok=True)
            df = create_sklearn_dataframe(data_list, page_map, config)
            if df is not None:
                csv_path = sklearn_dir / f"{split_name}.csv"
                df.to_csv(csv_path, index=False)
                logging.info(f"Saved Sklearn {split_name} dataset to '{csv_path}'")
    
    logging.info("Dataset creation complete.")
    
if __name__ == "__main__":
    main()