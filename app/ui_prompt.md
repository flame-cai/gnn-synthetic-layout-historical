As an expert in web development with javascript, vue.js as frontend and python (pytorch, torch-geometric) as backend please help me make improvements to my python application which allows the user to perfrom semi-autonomous layout analysis of historical manuscript images using graph neural networks (GNNs). The problem formulation is as follows:

Each character in the manuscript image is represented as a node in a graph. This is done using CRAFT, which detects character locations in the image by generating a heatmap of character locations. These character locations are used as node features (along with other features)

The applications currently processes the heatmap to create a graph representation of the manuscript page, to perfrom the task of text-line segmentation. However, in the future we may want to extend this to other layout analysis tasks such as textbox segmentation, reading order detection etc.

To perfrom text-line segmentation, the tool uses a GNN to label nodes (characters) belonging to the same text-line with the same label. The output of the GNN is then used to segment text-line images and create PAGE XML files for the manuscript page.
The current pipeline is as follows:
1) Resize the original images
2) Convert the images to heatmaps using CRAFT.
3) Convert the heatmap into a GNN friendly format with nodes as character locations (x,y) and font_size.
	Hence the gnn-dataset at his stage looks like:
  {page_id}_dims.txt containing:
	1250.0 642.0


	{page_id}_inputs_normalized.txt containing the normalized (x,y) coordinates of each character node and other features.
	0.455200 0.158400 0.017600
	0.592000 0.158400 0.011200
	0.392800 0.159200 0.008800
	0.430400 0.159200 0.010400
	0.447200 0.159200 0.013600
	
	{page_id}_inputs_unnormalized.txt containing the unnormalized (x,y) coordinates of each character node and other features.
	569.000000 198.000000 22.000000
	740.000000 198.000000 14.000000
	491.000000 199.000000 11.000000
	538.000000 199.000000 13.000000
	559.000000 199.000000 17.000000

4) Then a GNN pipeline is used label points (nodes) belonging to the same text-line with the same label. Hence the GNN pipeline creates a {page_id}_labels_textline.txt file.

	{page_id}_labels_textline.txt containing:
	0
	0
	0
	0
	0

	In this example, all 5 points belong to the same line hence have the same label 0.

However a problem is that some times the predicted labels in {page_id}_labels_textline.txt are incorrect. Hence we need to perfom human supervision after step 4, such that a human a manually verify if the predictions are correct, and if they are not, the human should be able to correct them. This is where the frontend comes in, allowing the user to hover over characters (nodes) to add edges or delete edges between them - the goal being to ensure that all characters (nodes) belonging to the same text-line are connected together.

The application frontend also allows to label textboxes, where hovering over text-lines marks them as belonging to the same textbox. (all text-lines belonging to the same textbox have same label). Hence all nodes (characters) belonging to text-lines in the same textbox should have the same textbox label.

{page_id}_labels_textbox.txt containing:
0
0
0
0
0

The application currently does not generate {page_id}_labels_textbox.txt, but this is a planned feature.


Hence the user flow is follows:
1) On the main page, the user uploads the manuscript images to be processed. The main page also allows the user to modify the default dimensions used to resize the original images in step 1 above (default longest side=2500).
2) After uploading, the backend processes the images till step 4 above, and then serves the frontend a page where the user can see the manuscript image overlayed with the detected character points (nodes) and edges between them. The frontend for page 1 should be displayed once the GNN inference (till step 4) is complete for that page, while the GNN inference for other pages can continue in the background. Handle this carefully to ensure smooth user experience.
3) Using the frontend, the user can then verify if the node labels are correct. If they are not correct, the user can correct them manually. In the frontend the user can also optionally choose to use the heuristic algorithm. The application also allows the user to label textboxes by hovering over text-lines to group them into textboxes is the textbox labelling mode.
4) Then the user can click a "Save and Proceed" button, which saves the corrected node labels to {page_id}_labels_textline.txt and then the backend code proceeds to step 5 above, generating the PAGE XML files and segmented line images for that page. The frontend then displays the next page for verification, while the backend continues processing other pages in the background.
5) This continues till all pages are processed.

Please study the above problem formulation and the current codebase carefully, and understand the entire flow. Then, please help me make the following changes to the application:

1) TODO: The frontend already allows textbox labelling, but the backend does not yet support saving textbox labels to {page_id}_labels_textbox.txt. Please add this functionality to the backend, ensuring that when the user clicks "Save and Proceed", the textbox labels are also saved correctly to {page_id}_labels_textbox.txt. Note that this is a nuanced change, as the textbox labels are at a higher level than text-line labels. So we will need to generate the PAGE XML files accordingly, ensuring that text-lines are grouped into textboxes correctly based on the textbox labels. Also right now the text-line reading order is determined using a simple top-to-bottom left-to-right heuristic. Please change this to ensure that this reading order is now applied inside each textbox, and the textboxes themselves are ordered top-to-bottom left-to-right. If the user has not labelled any textboxes, then all text-lines should be treated as belonging to a single textbox (as is the current behaviour).

2) TODO: The main page should allow the user to tweak the hyperparameter min_distance=20, along with the image resize dimension (longest side=2500). Please ensure that these hyperparameters are used correctly in the backend processing pipeline.

3) TODO: Add a button "Export PAGE XMLs" on the annotation page, which allows the user to download all generated PAGE XML files, and segmented line images as a zip file. This should include all PAGE XML files and segmented line images generated _so far_ (including the current page), even if the user has not yet completed the annotation for all (later) pages.


Do not make unnecessary changes to the gnn inference pipeline. 
IMPORTANT: Please make precise changes, and only tell me which functions or code blocks to replace/add/delete. Only make necessary changes. Ensure that the code is clean, modular and well documented. Follow best practices.

Please find the code below:


app
Tue Jan 13 10:41:39 AM IST 2026

# Complete Repository Structure:
# (showing all directories and files with token counts)
#/ (~16234 tokens)
#  └── app.py (~1365 tokens)
#  └── environment.yaml (~140 tokens)
#  └── gnn_inference.py (~7189 tokens)
#  └── inference.py (~1225 tokens)
#  └── README.md (~139 tokens)
#  └── segment_from_point_clusters.py (~6176 tokens)
#  /gnn_data_preparation/ (~10001 tokens)
#    └── config_models.py (~1087 tokens)
#    └── dataset_generator.py (~1011 tokens)
#    └── feature_engineering.py (~856 tokens)
#    └── graph_constructor.py (~4263 tokens)
#    └── __init__.py (~0 tokens)
#    └── main_create_dataset.py (~2553 tokens)
#    └── utils.py (~231 tokens)
#  /gnn_training/ (~0 tokens)
#    /gnn_training/training/ (~12007 tokens)
#      └── engine.py (~1872 tokens)
#      └── __init__.py (~0 tokens)
#      └── main_train_eval.py (~5147 tokens)
#      └── metrics.py (~2267 tokens)
#      └── utils.py (~1046 tokens)
#      └── visualization.py (~1675 tokens)
#      /gnn_training/training/models/ (~3321 tokens)
#        └── gnn_models.py (~2990 tokens)
#        └── __init__.py (~0 tokens)
#        └── sklearn_models.py (~331 tokens)
#  /input_manuscripts/ (~0 tokens)
#  /my-app/ (~1021 tokens)
#    └── .env (~10 tokens)
#    └── env.d.ts (~9 tokens)
#    └── .gitignore (~92 tokens)
#    └── index.html (~82 tokens)
#    └── package.json (~184 tokens)
#    └── README.md (~341 tokens)
#    └── tsconfig.app.json (~72 tokens)
#    └── tsconfig.json (~34 tokens)
#    └── tsconfig.node.json (~103 tokens)
#    └── vite.config.ts (~94 tokens)
#    /my-app/public/ (~0 tokens)
#    /my-app/src/ (~937 tokens)
#      └── App.vue (~884 tokens)
#      └── main.ts (~53 tokens)
#      /my-app/src/components/ (~8886 tokens)
#        └── ManuscriptViewer.vue (~8886 tokens)
#      /my-app/src/layout-analysis-utils/ (~2567 tokens)
#        └── LayoutGraphGenerator.js (~2567 tokens)
#      /my-app/src/router/ (~46 tokens)
#        └── index.ts (~46 tokens)
#      /my-app/src/stores/ (~76 tokens)
#        └── counter.ts (~76 tokens)
#  /pretrained_gnn/ (~960 tokens)
#    └── gnn_preprocessing_v2.yaml (~960 tokens)
#  /segmentation/ (~4979 tokens)
#    └── craft.py (~2654 tokens)
#    └── segment_graph.py (~2018 tokens)
#    └── utils.py (~307 tokens)
#    /segmentation/pretrained_unet_craft/ (~47 tokens)
#      └── README.md (~47 tokens)
#
---
---
app.py
---
# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import shutil
from pathlib import Path
import base64
import json

# Import your existing pipelines
from inference import process_new_manuscript
from gnn_inference import run_gnn_prediction_for_page, generate_xml_and_images_for_page
from segmentation.utils import load_images_from_folder

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './input_manuscripts'
MODEL_CHECKPOINT = "./pretrained_gnn/v2.pt"
DATASET_CONFIG = "./pretrained_gnn/gnn_preprocessing_v2.yaml"

@app.route('/upload', methods=['POST'])
def upload_manuscript():
    """
    Step 1 & 2: Upload images, resize (inference.py), Generate Heatmaps & GNN inputs.
    """
    manuscript_name = request.form.get('manuscriptName', 'default_manuscript')
    longest_side = int(request.form.get('longestSide', 2500))
    # Note: min_distance logic is embedded in segment_graph.py called by inference.py.
    # To expose it dynamically, you might need to modify inference.py to accept it as an arg.
    
    manuscript_path = os.path.join(UPLOAD_FOLDER, manuscript_name)
    images_path = os.path.join(manuscript_path, "images")
    
    if os.path.exists(manuscript_path):
        shutil.rmtree(manuscript_path)
    os.makedirs(images_path)

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    for file in files:
        if file.filename:
            file.save(os.path.join(images_path, file.filename))

    try:
        # Run Step 1-3: Resize and Generate Heatmaps/Points
        # You need to modify inference.py process_new_manuscript to accept target_longest_side
        # For now, we assume you modified it or we monkey-patch defaults
        process_new_manuscript(manuscript_path) 
        
        # Get list of processed pages
        processed_pages = []
        for f in sorted(Path(manuscript_path).glob("gnn-dataset/*_dims.txt")):
            processed_pages.append(f.name.replace("_dims.txt", ""))
            
        return jsonify({"message": "Processed successfully", "pages": processed_pages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/manuscript/<name>/pages', methods=['GET'])
def get_pages(name):
    manuscript_path = Path(UPLOAD_FOLDER) / name / "gnn-dataset"
    if not manuscript_path.exists():
        return jsonify([]), 404
    
    pages = sorted([f.name.replace("_dims.txt", "") for f in manuscript_path.glob("*_dims.txt")])
    return jsonify(pages)

@app.route('/semi-segment/<manuscript>/<page>', methods=['GET'])
def get_page_prediction(manuscript, page):
    """
    Step 4 Inference: Run GNN, get graph, return to frontend.
    """
    print("Received request for manuscript:", manuscript, "page:", page)
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    try:
        # Run GNN Inference
        graph_data = run_gnn_prediction_for_page(
            str(manuscript_path), 
            page, 
            MODEL_CHECKPOINT, 
            DATASET_CONFIG
        )
        
        # Load Image to send to frontend
        img_path = manuscript_path / "images_resized" / f"{page}.jpg"
        # if not img_path.exists():
        #      # Fallback if original wasn't resized
        #      img_path = manuscript_path / "images" / f"{page}.jpg"
             
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        response = {
            "image": encoded_string,
            "dimensions": graph_data['dimensions'],
            "points": [[n['x'], n['y']] for n in graph_data['nodes']],
            "graph": graph_data,
            "textline_labels": graph_data['textline_labels']
        }
        return jsonify(response)
        
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/semi-segment/<manuscript>/<page>', methods=['POST'])
def save_correction(manuscript, page):
    """
    Step 5: Receive corrected labels, Save, Generate XML/Lines.
    """
    data = request.json
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    
    # Extract data from frontend
    # The frontend sends 'textlineLabels' (list of ints) and 'graph' (edges with labels)
    textline_labels = data.get('textlineLabels')
    graph_data = data.get('graph')
    
    if not textline_labels or not graph_data:
        return jsonify({"error": "Missing labels or graph data"}), 400

    try:
        result = generate_xml_and_images_for_page(
            str(manuscript_path),
            page,
            textline_labels,
            graph_data['edges'],
            { # Pass default hyperparameters or read from request
                'BINARIZE_THRESHOLD': 0.5098,
                'BBOX_PAD_V': 0.7,
                'BBOX_PAD_H': 0.5,
                'CC_SIZE_THRESHOLD_RATIO': 0.4
            }
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/save-graph/<manuscript>/<page>', methods=['POST'])
def save_generated_graph(manuscript, page):
    # Optional helper if you want to save the heuristic graph without generating XML immediately
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
---
environment.yaml
---
name: gnn_layout

channels:
  - conda-forge
  - pytorch
  - pyg
  - nvidia
  - defaults

dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - scikit-image
  - matplotlib
  - pytorch
  - torchvision
  - pytorch-cuda=12.1 
  - pyg
  - flask
  - flask-cors
  - flask-sqlalchemy
  - werkzeug
  - pillow
  - opencv
  - python-dotenv
  - packaging
  - six
  - natsort
  - pyyaml
  - conda-forge::pytorch_spline_conv
  - pip:
    - shapely
    - lmdb
    - nltk
    - python-json-logger
    - regex
    - pydantic
    - wandb
    
---
gnn_data_preparation/config_models.py
---
# data_creation/config_models.py
from pydantic import BaseModel, Field, conint, confloat
from typing import List, Literal, Optional, Annotated


# Feature Engineering Models
class HeuristicDegreeEncoding(BaseModel):
    linear_map_factor: float = 0.1
    one_hot_max_degree: int = 10

class OverlapEncoding(BaseModel):
    linear_map_factor: float = 0.1
    one_hot_max_overlap: int = 10

class FeaturesConfig(BaseModel):
    use_node_coordinates: bool = True
    use_node_font_size: bool = True
    use_heuristic_degree: bool = True
    heuristic_degree_encoding: Literal["linear_map", "one_hot"] = "linear_map"
    heuristic_degree_encoding_params: HeuristicDegreeEncoding = Field(default_factory=HeuristicDegreeEncoding)
    use_relative_distance: bool = True
    use_euclidean_distance: bool = True
    use_aspect_ratio_rel: bool = True
    use_overlap: bool = True
    overlap_encoding: Literal["linear_map", "one_hot"] = "linear_map"
    overlap_encoding_params: OverlapEncoding = Field(default_factory=OverlapEncoding)
    use_page_aspect_ratio: bool = True


# Graph Construction Models
class HeuristicParams(BaseModel):
    k: int = 10
    cosine_sim_threshold: float = -0.8

# Define parameter models for each connectivity strategy
class KnnParams(BaseModel):
    k: int = 10

class SecondShortestHeuristicParams(BaseModel):
    k: int = 10
    cosine_sim_threshold: float = -0.8
    min_angle_degrees: float = 45.0 # Add this new parameter with a default value

class AngularKnnParams(BaseModel):
    k: int = 50  # K for angular KNN
    sector_angle_degrees: float = 20.0  # Minimum angle between edges to consider them connected

# Update ConnectivityConfig to handle a list of strategies and their params
class ConnectivityConfig(BaseModel):
    strategies: List[Literal["knn", "second_shortest_heuristic", "angular_knn"]] = []
    knn_params: Optional[KnnParams] = Field(default_factory=KnnParams)
    second_shortest_params: Optional[SecondShortestHeuristicParams] = Field(default_factory=SecondShortestHeuristicParams)
    angular_knn_params: Optional[AngularKnnParams] = Field(default_factory=AngularKnnParams)

class InputGraphConfig(BaseModel):
    use_heuristic_graph: bool = True
    heuristic_params: HeuristicParams = Field(default_factory=HeuristicParams)
    connectivity: ConnectivityConfig = Field(default_factory=ConnectivityConfig)
    directionality: Literal["bidirectional", "unidirectional"] = "bidirectional"



# Ground Truth Model
class GroundTruthConfig(BaseModel):
    algorithm: Literal["mst", "greedy_path"] = "mst"

# MODIFIED: Simplified SplittingConfig for the new fixed-split strategy
class SplittingConfig(BaseModel):
    """Configuration for splitting the validation/test dataset."""
    random_seed: int = 49
    val_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.75 # Ratio of the val/test data to be used for validation

# Sklearn Format Model
class NHopConfig(BaseModel):
    hops: int = 1
    aggregations: List[str] = ["mean", "std"]
    
class SklearnFormatConfig(BaseModel):
    enabled: bool = True
    features: List[str] = ["source_node_features", "target_node_features", "edge_features", "page_features"]
    use_n_hop_features: bool = False
    n_hop_config: Optional[NHopConfig] = Field(default_factory=NHopConfig)

# Top-level Configuration Model
class DatasetCreationConfig(BaseModel):
    """
    Top-level configuration model for the entire dataset creation process.
    """
    # make these optional in pydantic..
    version: Optional[str] = None
    manuscript_name: Optional[str] = None
    train_data_dir: Optional[str] = None
    val_test_data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    min_nodes_per_page: Annotated[int, Field(ge=1)] = 10
    
    # This field now uses the new, simplified SplittingConfig
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    
    # The rest of the configuration remains unchanged
    input_graph: InputGraphConfig = Field(default_factory=InputGraphConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)
    sklearn_format: SklearnFormatConfig = Field(default_factory=SklearnFormatConfig)
---
gnn_data_preparation/dataset_generator.py
---
# data_creation/dataset_generator.py
import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from typing import List, Dict

class HistoricalLayoutGNNDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for the historical layout analysis task.
    """
    def __init__(self, root, transform=None, pre_transform=None, data_list=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # We process manually, so this can be empty
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Data is assumed to be present locally in the specified raw dir
        pass

    def process(self):
        # If data is provided directly, just save it
        if self.data_list is not None:
            data, slices = self.collate(self.data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            # This would be used if we were processing from raw files inside this class
            # But we are doing it in the main script for better control over folds.
            print("Processing from raw files is not implemented directly in the class. "
                  "Please provide a data_list during initialization.")

def create_sklearn_dataframe(data_list: List[Data], page_map: Dict[int, str], config) -> pd.DataFrame:
    """
    Converts a list of PyG Data objects into a single pandas DataFrame
    suitable for training with scikit-learn models.
    """
    if not config.sklearn_format.enabled:
        return None

    all_edges_data = []
    
    node_feature_names = []
    if config.features.use_node_coordinates: node_feature_names.extend(['x', 'y'])
    if config.features.use_node_font_size: node_feature_names.append('font_size')
    # Add other node features names here if they are simple values
    
    for i, data in enumerate(data_list):
        page_id = page_map[i]
        edge_index = data.edge_index.T.numpy()
        edge_y = data.edge_y.numpy()
        
        for edge_idx, (u, v) in enumerate(edge_index):
            row = {'page_id': page_id, 'source_node_id': u, 'target_node_id': v}
            
            # Source and Target Node Features
            if "source_node_features" in config.sklearn_format.features:
                for feat_idx, name in enumerate(node_feature_names):
                    row[f'source_{name}'] = data.x[u, feat_idx].item()
            
            if "target_node_features" in config.sklearn_format.features:
                for feat_idx, name in enumerate(node_feature_names):
                    row[f'target_{name}'] = data.x[v, feat_idx].item()
            
            # Edge Features
            if "edge_features" in config.sklearn_format.features and data.edge_attr is not None:
                # This needs to be more specific based on config, but for now we dump them
                for feat_idx in range(data.edge_attr.shape[1]):
                    row[f'edge_attr_{feat_idx}'] = data.edge_attr[edge_idx, feat_idx].item()
                    
            # Page Features
            if "page_features" in config.sklearn_format.features and hasattr(data, 'page_aspect_ratio'):
                row['page_aspect_ratio'] = data.page_aspect_ratio.item()

            # Target Label
            row['label'] = edge_y[edge_idx]
            
            all_edges_data.append(row)

    df = pd.DataFrame(all_edges_data)
    # Note: N-hop features are complex and would require a separate graph traversal step here.
    # This is a placeholder for that advanced functionality.
    if config.sklearn_format.use_n_hop_features:
        print("Warning: N-hop feature generation for sklearn is an advanced feature and not yet implemented.")
        
    return df
---
gnn_data_preparation/feature_engineering.py
---
# data_creation/feature_engineering.py
import torch
import numpy as np

try:
    from .config_models import FeaturesConfig
except ImportError:
    from config_models import FeaturesConfig


def get_node_features(points: np.ndarray, heuristic_degrees: np.ndarray, config: FeaturesConfig) -> torch.Tensor:
    """Combines all enabled node features into a single tensor."""
    features_list = []
    
    if config.use_node_coordinates:
        features_list.append(torch.from_numpy(points[:, :2]).float())
    
    if config.use_node_font_size:
        features_list.append(torch.from_numpy(points[:, 2]).float().unsqueeze(1))
        
    if config.use_heuristic_degree:
        deg_feat = encode_categorical_feature(
            heuristic_degrees,
            config.heuristic_degree_encoding,
            config.heuristic_degree_encoding_params.linear_map_factor,
            config.heuristic_degree_encoding_params.one_hot_max_degree
        )
        features_list.append(deg_feat)
        
    return torch.cat(features_list, dim=1)


def get_edge_features(edge_index: torch.Tensor, node_features: torch.Tensor, heuristic_edge_counts: dict, config: FeaturesConfig) -> torch.Tensor:
    """Computes all enabled edge features for the given edges."""
    features_list = []
    source_nodes, target_nodes = edge_index[0], edge_index[1]
    
    # Use original node coordinates for distance calculations
    # Assuming first 2 features are x, y
    source_pos = node_features[source_nodes, :2]
    target_pos = node_features[target_nodes, :2]
    
    relative_pos = target_pos - source_pos
    
    if config.use_relative_distance:
        features_list.append(relative_pos) # rel_x, rel_y
        
    if config.use_euclidean_distance:
        dist = torch.linalg.norm(relative_pos, dim=1).unsqueeze(1)
        features_list.append(dist)
        
    if config.use_aspect_ratio_rel:
        # Add epsilon for stability
        aspect = torch.abs(relative_pos[:, 1]) / (torch.abs(relative_pos[:, 0]) + 1e-6)
        features_list.append(aspect.unsqueeze(1))
        
    if config.use_overlap:
        overlaps = []
        for u, v in edge_index.T.numpy():
            edge_key = tuple(sorted((u, v)))
            overlaps.append(heuristic_edge_counts.get(edge_key, 0))
        overlaps = np.array(overlaps)
        
        overlap_feat = encode_categorical_feature(
            overlaps,
            config.overlap_encoding,
            config.overlap_encoding_params.linear_map_factor,
            config.overlap_encoding_params.one_hot_max_overlap
        )
        features_list.append(overlap_feat)

    if not features_list:
        return None

    return torch.cat(features_list, dim=1)

def encode_categorical_feature(values: np.ndarray, method: str, factor: float, max_val: int) -> torch.Tensor:
    """Encodes a categorical feature using the specified method."""
    if method == "linear_map":
        return torch.from_numpy(values).float().unsqueeze(1) * factor
    elif method == "one_hot":
        # Clamp values to the max to avoid oversized tensors
        values_clamped = np.clip(values, 0, max_val)
        one_hot = torch.nn.functional.one_hot(torch.from_numpy(values_clamped).long(), num_classes=max_val + 1)
        return one_hot.float()
    else:
        raise ValueError(f"Unknown encoding method: {method}")
---
gnn_data_preparation/graph_constructor.py
---
# data_creation/graph_constructor.py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.neighbors import KDTree, NearestNeighbors
from itertools import combinations
from collections import Counter, defaultdict
import logging
from dataclasses import dataclass
from scipy.spatial import KDTree

try:
    # Case 1: run as part of a package
    from .config_models import InputGraphConfig, GroundTruthConfig, KnnParams, SecondShortestHeuristicParams
except ImportError:
    # Case 2: run directly (standalone script)
    from config_models import InputGraphConfig, GroundTruthConfig, KnnParams, SecondShortestHeuristicParams



@dataclass
class AngularKnnParams:
    """Parameters for the Angular K-Nearest Neighbor function."""
    # The width of each angular sector in degrees.
    sector_angle_degrees: float = 10.0
    
    # The number of candidate neighbors to check for each point.
    # This is a crucial performance parameter. A higher value is more accurate
    # for sparse sectors but slower.
    k: int = 50


def create_heuristic_graph(points: np.ndarray, page_dims: dict, config: InputGraphConfig) -> dict:
    """Creates a heuristic graph based on proximity and collinearity."""
    n_points = len(points)
    params = config.heuristic_params
    if n_points < 3: #params.k:
        logging.warning(f"Not enough points ({n_points}) to build heuristic graph with k={params.k}. Skipping.")
        return {"edges": set(), "degrees": np.zeros(n_points, dtype=int), "edge_counts": Counter()}

    # Normalize points for stable KDTree and cosine similarity
    normalized_points = np.copy(points)
    max_dim = max(page_dims['width'], page_dims['height'])
    normalized_points[:, :2] /= max_dim
    
    kdtree = KDTree(normalized_points[:, :2])
    heuristic_directed_edges = []
    
    for i in range(n_points):
        num_potential_neighbors = n_points - 1
        # We need at least 2 neighbors to form a pair.
        if num_potential_neighbors < 2:
            continue
        # The k for the query must be less than or equal to n_points.
        # We query for min(config_k, potential_neighbors) + 1 (for the point itself)
        k_for_query = min(params.k, num_potential_neighbors)
        _, neighbor_indices = kdtree.query(normalized_points[i, :2].reshape(1, -1), k=k_for_query + 1)
        
        # # Query k+1 to exclude the point itself
        # _, neighbor_indices = kdtree.query(normalized_points[i, :2].reshape(1, -1), k=params.k + 1)
        neighbor_indices = neighbor_indices[0][1:]
        
        best_pair, min_dist_sum = None, float('inf')
        
        # Find two neighbors that are most collinear and opposite to the current point
        for n1_idx, n2_idx in combinations(neighbor_indices, 2):
            vec1 = normalized_points[n1_idx, :2] - normalized_points[i, :2]
            vec2 = normalized_points[n2_idx, :2] - normalized_points[i, :2]
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0: continue
            
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            
            if cosine_sim < params.cosine_sim_threshold:
                dist_sum = norm1 + norm2
                if dist_sum < min_dist_sum:
                    min_dist_sum, best_pair = dist_sum, (n1_idx, n2_idx)
    
        if best_pair:
            heuristic_directed_edges.extend([(i, best_pair[0]), (i, best_pair[1])])

    # Convert to undirected edges for initial set and calculate degrees/overlaps
    heuristic_edges = {tuple(sorted(edge)) for edge in heuristic_directed_edges}
    degrees = np.zeros(n_points, dtype=int)
    for u, v in heuristic_edges:
        degrees[u] += 1
        degrees[v] += 1
        
    edge_counts = Counter(tuple(sorted(edge)) for edge in heuristic_directed_edges)

    return {"edges": heuristic_edges, "degrees": degrees, "edge_counts": edge_counts}


# Modify the function signature to accept its own specific parameters for better modularity
def add_knn_edges(points: np.ndarray, existing_edges: set, params: KnnParams) -> set:
    """Adds K-Nearest Neighbor edges to the graph."""
    n_points = len(points)
    k = params.k
    if n_points <= k:
        k = n_points - 1

    if k <= 0:
        return set()

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(points[:, :2])
    _, indices = nn.kneighbors(points[:, :2])

    knn_edges = set()
    for i in range(n_points):
        for j_idx in indices[i, 1:]:  # Skip self
            edge = tuple(sorted((i, j_idx)))
            if edge not in existing_edges:
                knn_edges.add(edge)
    return knn_edges

def add_angular_knn_edges(points: np.ndarray, existing_edges: set, params: AngularKnnParams) -> set:
    """
    Adds edges to the nearest neighbor in distinct angular sectors around each point.

    Instead of finding the k absolute nearest neighbors, this method divides the
    360-degree space around each point into sectors (e.g., 10-degree cones)
    and finds the single closest neighbor within each sector. This ensures a
    more directionally uniform set of connections.

    Args:
        points: A NumPy array of shape (n_points, D) where D >= 2.
        existing_edges: A set of existing edges (as sorted tuples) to avoid duplication.
        params: An AngularKnnParams object with configuration.

    Returns:
        A set of new, undirected edges represented as sorted tuples.
    """
    n_points = len(points)
    if n_points < 2:
        return set()

    # --- Parameter Validation and Setup ---
    if not (0 < params.sector_angle_degrees <= 180):
        logging.error("sector_angle_degrees must be between 0 and 180.")
        return set()
    
    num_sectors = int(360 / params.sector_angle_degrees)
    
    # To avoid missing neighbors in sparse regions, we must check a reasonable
    # number of candidate points. We cap this at the total number of other points.
    k_for_query = min(params.k, n_points - 1)
    if k_for_query <= 0:
        return set()

    # --- Algorithm Implementation ---
    
    # 1. Build a spatial index (KDTree) for all points. This is the crucial
    #    first step for achieving high performance.
    kdtree = KDTree(points[:, :2])
    
    new_edges = set()

    # 2. Iterate through each point to find its angular neighbors.
    for i in range(n_points):
        # 3. Fast Pre-selection: Query the KDTree for a set of candidate neighbors.
        #    This is vastly more efficient than checking all N-1 other points.
        #    We query for k+1 because the point itself is always the first neighbor.
        _, candidate_indices = kdtree.query(points[i, :2], k=k_for_query + 1)
        
        # Exclude the point itself (which is always at index 0)
        candidate_indices = candidate_indices[1:]
        
        # --- Vectorized Calculations on Candidates ---
        
        # 4. Calculate vectors from the current point 'i' to all its candidates.
        vectors = points[candidate_indices, :2] - points[i, :2]
        
        # 5. Calculate the distances (norms) of these vectors.
        distances = np.linalg.norm(vectors, axis=1)
        
        # 6. Calculate the angles of these vectors in degrees [0, 360).
        #    np.arctan2 is used for quadrant-aware angle calculation.
        angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles_deg = np.rad2deg(angles_rad) % 360
        
        # 7. Determine the angular sector index for each candidate.
        sector_indices = np.floor(angles_deg / params.sector_angle_degrees).astype(int)
        
        # --- Find the Best Neighbor in Each Sector ---
        
        # 8. We now efficiently find the closest point within each sector.
        #    Initialize arrays to store the minimum distance and corresponding neighbor
        #    index found so far for each sector.
        min_dists_in_sector = np.full(num_sectors, np.inf)
        best_neighbor_in_sector = np.full(num_sectors, -1, dtype=int)
        
        # This loop is fast as it only iterates over the small set of 'k candidates'.
        for j in range(len(candidate_indices)):
            sector_idx = sector_indices[j]
            dist = distances[j]
            
            if dist < min_dists_in_sector[sector_idx]:
                min_dists_in_sector[sector_idx] = dist
                best_neighbor_in_sector[sector_idx] = candidate_indices[j]

        # 9. Create new edges from the results.
        for neighbor_idx in best_neighbor_in_sector:
            if neighbor_idx != -1:  # A neighbor was found in this sector
                edge = tuple(sorted((i, neighbor_idx)))
                if edge not in existing_edges:
                    new_edges.add(edge)
                    
    return new_edges

def add_second_shortest_heuristic_edges(points: np.ndarray, page_dims: dict, existing_edges: set, params: SecondShortestHeuristicParams) -> set:
    """
    Creates edges based on a secondary neighbor pair that is angularly separated
    from the primary (shortest) pair.
    """
    n_points = len(points)
    if n_points < 4:  # Need at least 3 neighbors for a chance at two pairs
        logging.debug(f"Not enough points ({n_points}) for second_shortest_heuristic. Skipping.")
        return set()
    
    # Pre-calculate the cosine of the minimum angle for efficiency.
    # We use the absolute value of the cosine for the check.
    cos_angle_threshold = np.cos(np.deg2rad(params.min_angle_degrees))

    # Normalize points for stable KDTree and cosine similarity
    normalized_points = np.copy(points)
    max_dim = max(page_dims['width'], page_dims['height'])
    normalized_points[:, :2] /= max_dim

    kdtree = KDTree(normalized_points[:, :2])
    new_directed_edges = []

    for i in range(n_points):
        num_potential_neighbors = n_points - 1
        if num_potential_neighbors < 3:
            continue

        k_for_query = min(params.k, num_potential_neighbors)
        _, neighbor_indices = kdtree.query(normalized_points[i, :2].reshape(1, -1), k=k_for_query + 1)
        neighbor_indices = neighbor_indices[0][1:]

        valid_pairs = []
        for n1_idx, n2_idx in combinations(neighbor_indices, 2):
            vec1 = normalized_points[n1_idx, :2] - normalized_points[i, :2]
            vec2 = normalized_points[n2_idx, :2] - normalized_points[i, :2]
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0: continue

            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)

            if cosine_sim < params.cosine_sim_threshold:
                dist_sum = norm1 + norm2
                valid_pairs.append({'dist': dist_sum, 'pair': (n1_idx, n2_idx)})

        # We need at least two valid pairs to proceed
        if len(valid_pairs) < 2:
            continue

        # Sort pairs by distance to find the best and subsequent candidates
        valid_pairs.sort(key=lambda x: x['dist'])
        
        best_pair_nodes = valid_pairs[0]['pair']
        
        # Define the direction vector for the best pair
        # This vector points from the center node 'i' to the midpoint of its pair
        best_pair_midpoint = (normalized_points[best_pair_nodes[0], :2] + normalized_points[best_pair_nodes[1], :2]) / 2
        vec_best = best_pair_midpoint - normalized_points[i, :2]
        norm_vec_best = np.linalg.norm(vec_best)

        if norm_vec_best == 0: continue

        # Find the first subsequent pair that meets the angle criteria
        for candidate in valid_pairs[1:]:
            candidate_nodes = candidate['pair']
            
            # Define the direction vector for the candidate pair
            candidate_midpoint = (normalized_points[candidate_nodes[0], :2] + normalized_points[candidate_nodes[1], :2]) / 2
            vec_candidate = candidate_midpoint - normalized_points[i, :2]
            norm_vec_candidate = np.linalg.norm(vec_candidate)

            if norm_vec_candidate == 0: continue

            # Calculate the cosine of the angle between the two direction vectors
            cos_angle_between_pairs = np.dot(vec_best, vec_candidate) / (norm_vec_best * norm_vec_candidate)

            # Check if the angle is large enough (i.e., vectors are not aligned)
            # abs(cos(theta)) < cos(45) means theta is between 45 and 135 degrees.
            if abs(cos_angle_between_pairs) < cos_angle_threshold:
                # Found a suitable "cross" pair, add its edges and stop searching for this point `i`
                second_best_pair = candidate_nodes
                new_directed_edges.extend([(i, second_best_pair[0]), (i, second_best_pair[1])])
                break # Move to the next point i

    # Convert to undirected edges, ensuring no duplicates are added
    new_edges = set()
    for u, v in new_directed_edges:
        edge = tuple(sorted((u, v)))
        if edge not in existing_edges:
            new_edges.add(edge)
            
    return new_edges



# Update the main graph creation function to iterate through the chosen strategies
def create_input_graph_edges(points: np.ndarray, page_dims: dict, config: InputGraphConfig) -> dict:
    """Constructs the full input graph by combining heuristic and connectivity strategies."""
    heuristic_result = {"edges": set(), "degrees": np.zeros(len(points), dtype=int), "edge_counts": Counter()}
    if config.use_heuristic_graph:
        heuristic_result = create_heuristic_graph(points, page_dims, config)

    # Union of all edges from different strategies
    all_edges = heuristic_result["edges"].copy()

    # Iterate through the list of selected strategies and add their edges
    for strategy in config.connectivity.strategies:
        current_edges = all_edges.copy() # Pass a copy to avoid modifying the set while iterating
        if strategy == "knn" and config.connectivity.knn_params:
            knn_edges = add_knn_edges(points, current_edges, config.connectivity.knn_params)
            all_edges.update(knn_edges)
        elif strategy == "second_shortest_heuristic" and config.connectivity.second_shortest_params:
            second_shortest_edges = add_second_shortest_heuristic_edges(points, page_dims, current_edges, config.connectivity.second_shortest_params)
            all_edges.update(second_shortest_edges)
        elif strategy == "angular_knn" and config.connectivity.angular_knn_params:
            angular_knn_edges = add_angular_knn_edges(points, current_edges, config.connectivity.angular_knn_params)
            all_edges.update(angular_knn_edges)
    
    return {
        "edges": all_edges,
        "heuristic_degrees": heuristic_result["degrees"],
        "heuristic_edge_counts": heuristic_result["edge_counts"]
    }

def create_ground_truth_graph_edges(points: np.ndarray, labels: np.ndarray, config: GroundTruthConfig) -> set:
    """Constructs the ground truth graph by connecting nodes within the same textline."""
    gt_edges = set()
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1: continue # Skip noise/unlabeled points
            
        indices = np.where(labels == label)[0]
        if len(indices) < 2: continue

        line_points = points[indices, :2]
        
        if config.algorithm == "mst":
            # Build a distance matrix and run MST
            dist_matrix = np.linalg.norm(line_points[:, np.newaxis, :] - line_points[np.newaxis, :, :], axis=2)
            csr_dist = csr_matrix(dist_matrix)
            mst = minimum_spanning_tree(csr_dist)
            
            # Convert MST to edges
            rows, cols = mst.nonzero()
            for i, j in zip(rows, cols):
                u, v = indices[i], indices[j]
                gt_edges.add(tuple(sorted((u, v))))

        elif config.algorithm == "greedy_path":
            # A simple greedy path construction
            # This can be implemented as an alternative
            # For now, we focus on MST
            logging.warning("Greedy Path GT construction not implemented, falling back to MST.")
            # Build a distance matrix and run MST
            dist_matrix = np.linalg.norm(line_points[:, np.newaxis, :] - line_points[np.newaxis, :, :], axis=2)
            csr_dist = csr_matrix(dist_matrix)
            mst = minimum_spanning_tree(csr_dist)
            
            # Convert MST to edges
            rows, cols = mst.nonzero()
            for i, j in zip(rows, cols):
                u, v = indices[i], indices[j]
                gt_edges.add(tuple(sorted((u, v))))

    return gt_edges
---
gnn_data_preparation/__init__.py
---

---
gnn_data_preparation/main_create_dataset.py
---
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
---
gnn_data_preparation/utils.py
---
# data_creation/utils.py
import logging
import sys
from typing import List
from pathlib import Path

def setup_logging(log_path: Path):
    """Sets up a logger that prints to console and saves to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

def find_page_ids(data_dir: Path) -> List[str]:
    """Finds all unique page identifiers in the raw data directory."""
    page_ids = set()
    for f in data_dir.glob('*_dims.txt'):
        page_id = f.name.replace('_dims.txt', '')
        page_ids.add(page_id)
    
    sorted_ids = sorted(list(page_ids))
    logging.info(f"Found {len(sorted_ids)} pages in '{data_dir}'.")
    return sorted_ids
---
gnn_inference.py
---
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

# --- STAGE 1: Inference / Loading ---
def run_gnn_prediction_for_page(manuscript_path, page_id, model_path, config_path):
    """
    Retrieves graph data for a page. 
    Priority:
    1. Check 'layout_analysis_output/gnn-format/' for saved manual corrections (_edges.txt).
    2. If not found, run GNN inference from scratch using 'gnn-dataset/'.
    """
    print(f"Fetching data for page: {page_id}")
    
    # Define Directories
    base_path = Path(manuscript_path)
    raw_input_dir = base_path / "gnn-dataset"               # Folder A (Inputs)
    history_dir = base_path / "layout_analysis_output" / "gnn-format" # Folder B (Saved Corrections)
    
    # 1. Load Essential Node Data (Always required)
    file_path = raw_input_dir / f"{page_id}_inputs_normalized.txt"
    dims_path = raw_input_dir / f"{page_id}_dims.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data for page {page_id} not found in {raw_input_dir}")

    points_normalized = np.loadtxt(file_path)
    if points_normalized.ndim == 1: 
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
        "dimensions": [full_width, full_height]
    }

    # 2. Check for Saved Corrections
    saved_edges_path = history_dir / f"{page_id}_edges.txt"
    saved_labels_path = history_dir / f"{page_id}_labels_textline.txt"
    
    if saved_edges_path.exists():
        print(f"Found saved corrections for {page_id}. Loading from disk...")
        
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
            print(f"Warning: Error reading edges file: {e}. Returning empty edges.")
            
        response["edges"] = saved_edges
        
        if saved_labels_path.exists():
            try:
                labels = np.loadtxt(saved_labels_path, dtype=int)
                if labels.size == len(points_normalized):
                     response["textline_labels"] = labels.tolist()
            except Exception:
                pass 
        
        return response

    # 3. No Saved State -> Run GNN Inference
    print(f"No saved state found. Running GNN Inference...")
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


# --- STAGE 2: Generation / Saving ---
def generate_xml_and_images_for_page(manuscript_path, page_id, node_labels, graph_edges, args_dict):
    """
    Saves user corrections and regenerates XML.
    Critical Fix: Accepts all edges in 'graph_edges' list as valid, ignoring specific label values 
    to accommodate manually added edges (which might have label=0).
    """
    
    base_path = Path(manuscript_path)
    raw_input_dir = base_path / "gnn-dataset"
    output_dir = base_path / "layout_analysis_output"
    gnn_format_dir = output_dir / "gnn-format"
    gnn_format_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Save Corrected Edges to .txt
    # ---------------------------------------------------------
    unique_edges = set()
    
    # FIX: Iterate over all provided edges. 
    # If it is in the list, it implies it exists in the graph.
    # We strip 'label' checks or allow 0 and 1.
    for e in graph_edges:
        # Check source/target keys existence for safety
        if 'source' in e and 'target' in e:
            u, v = sorted((int(e['source']), int(e['target'])))
            unique_edges.add((u, v))
            
    edges_save_path = gnn_format_dir / f"{page_id}_edges.txt"
    if unique_edges:
        np.savetxt(edges_save_path, list(unique_edges), fmt='%d')
    else:
        open(edges_save_path, 'w').close()
        
    print(f"Saved {len(unique_edges)} edges to {edges_save_path}")

    # ---------------------------------------------------------
    # 2. Copy Input Files (Redundancy for Segmentation Script)
    # ---------------------------------------------------------
    for suffix in ["_inputs_normalized.txt", "_inputs_unnormalized.txt", "_dims.txt"]:
        src = raw_input_dir / f"{page_id}{suffix}"
        dst = gnn_format_dir / f"{page_id}{suffix}"
        if src.exists():
            shutil.copy(src, dst)

    # ---------------------------------------------------------
    # 3. Calculate Structural Labels
    # ---------------------------------------------------------
    num_nodes = len(node_labels)
    
    if unique_edges:
        row, col = zip(*unique_edges)
        all_rows = list(row) + list(col)
        all_cols = list(col) + list(row)
        data = np.ones(len(all_rows))
        adj = csr_matrix((data, (all_rows, all_cols)), shape=(num_nodes, num_nodes))
    else:
        adj = csr_matrix((num_nodes, num_nodes))

    n_components, final_structural_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    
    labels_save_path = gnn_format_dir / f"{page_id}_labels_textline.txt"
    np.savetxt(labels_save_path, final_structural_labels, fmt='%d')

    # ---------------------------------------------------------
    # 4. Run Segmentation & XML Generation
    # ---------------------------------------------------------
    unnorm_path = raw_input_dir / f"{page_id}_inputs_unnormalized.txt"
    points_unnormalized = np.loadtxt(unnorm_path)
    if points_unnormalized.ndim == 1: 
        points_unnormalized = points_unnormalized.reshape(1, -1)
    
    dims_path = raw_input_dir / f"{page_id}_dims.txt"
    dims = np.loadtxt(dims_path)
    page_dims = {'width': dims[0], 'height': dims[1]}

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
    
    from gnn_inference import create_page_xml 
    
    create_page_xml(
        page_id,
        unique_edges,
        points_unnormalized,
        page_dims,
        xml_output_dir / f"{page_id}.xml",
        final_structural_labels, 
        polygons_data,
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



def create_page_xml(
    page_id,
    model_positive_edges,
    points_unnormalized,
    page_dims,
    output_path: Path,
    pred_node_labels: np.ndarray,
    polygons_data: dict,
    use_best_fit_line: bool = False,
    extend_percentage: float = 0.01,
    image_path: Path = None, 
    save_vis: bool = True
):
    """
    Generates a PAGE XML file and optionally saves a visualization of the 
    polygons and baselines overlayed on the image.
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

    # Define dimensions (Your logic uses *2 scaling for the XML)
    final_w = int(page_dims['width'] * 2)
    final_h = int(page_dims['height'] * 2)

    page = ET.SubElement(pc_gts, "Page", attrib={
        "imageFilename": f"{page_id}.jpg",
        "imageWidth": str(final_w),
        "imageHeight": str(final_h)
    })

    # --- VISUALIZATION SETUP ---
    vis_img = None
    if save_vis:
        # Try to load original image to overlay
        if image_path and image_path.exists():
            vis_img = cv2.imread(str(image_path))
            # Ensure visualization matches XML coordinate space
            if vis_img is not None:
                if vis_img.shape[0] != final_h or vis_img.shape[1] != final_w:
                    vis_img = cv2.resize(vis_img, (final_w, final_h))
        
        # Fallback to black canvas if image not found or load failed
        if vis_img is None:
            vis_img = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    # ---------------------------

    min_x = np.min(points_unnormalized[:, 0])
    min_y = np.min(points_unnormalized[:, 1])
    max_x = np.max(points_unnormalized[:, 0])
    max_y = np.max(points_unnormalized[:, 1])
    
    # Region logic (scaled by 2)
    region_coords_str = f"{int(min_x*2)},{int(min_y*2)} {int(max_x*2)},{int(min_y*2)} {int(max_x*2)},{int(max_y*2)} {int(min_x*2)},{int(max_y*2)}"

    text_region = ET.SubElement(page, "TextRegion", id="region_1")
    ET.SubElement(text_region, "Coords", points=region_coords_str)

    # Visualize Region Boundary (Yellow)
    if save_vis and vis_img is not None:
        pts = np.array([[int(min_x*2), int(min_y*2)], [int(max_x*2), int(min_y*2)], 
                        [int(max_x*2), int(max_y*2)], [int(min_x*2), int(max_y*2)]], np.int32)
        cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

    for component in components:
        if not component: continue
        
        component_points = np.array([points_unnormalized[idx] for idx in component])
        if len(component_points) < 1:
            continue

        line_label = pred_node_labels[component[0]]
        text_line = ET.SubElement(text_region, "TextLine", id=f"line_{line_label + 1}")
        
        # --- CALCULATE BASELINE ---
        baseline_points_str = ""
        baseline_vis_points = [] # To store points for visualization

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
                # XML points (scaled by 2)
                x1, y1 = int(p1[0] * 2), int(p1[1] * 2)
                x2, y2 = int(p2[0] * 2), int(p2[1] * 2)
                baseline_points_str = f"{x1},{y1} {x2},{y2}"
                baseline_vis_points = [[x1, y1], [x2, y2]]
            else:
                continue
        else:
            path_indices = trace_component_with_backtracking(component, adj)
            if len(path_indices) < 1: continue
            ordered_points = [points_unnormalized[idx] for idx in path_indices]
            
            # Format for XML
            baseline_points_str = " ".join([f"{int(p[0]*2)},{int((p[1]+(p[2]/2))*2)}" for p in ordered_points])
            
            # Format for Visualizer
            baseline_vis_points = [[int(p[0]*2), int((p[1]+(p[2]/2))*2)] for p in ordered_points]

        ET.SubElement(text_line, "Baseline", points=baseline_points_str)
        
        # --- HANDLE POLYGON COORDS ---
        polygon_vis_points = []
        if line_label in polygons_data:
            polygon_points = polygons_data[line_label]
            # XML expects string "x,y x,y ..."
            coords_str = " ".join([f"{p[0]},{p[1]}" for p in polygon_points]) 
            ET.SubElement(text_line, "Coords", points=coords_str)
            
            # Store for visualization (already in correct scale per your comment)
            polygon_vis_points = polygon_points
        else:
            logging.warning(f"Page {page_id}: No polygon data found for line label {line_label}, Coords tag will be omitted.")

        # --- DRAW ON VISUALIZATION ---
        if save_vis and vis_img is not None:
            # 1. Draw Polygon (Green)
            if len(polygon_vis_points) > 0:
                poly_pts = np.array(polygon_vis_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Draw ID at the start of the polygon
                cv2.putText(vis_img, str(line_label + 1), tuple(poly_pts[0][0]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 2. Draw Baseline (Red)
            if len(baseline_vis_points) > 0:
                base_pts = np.array(baseline_vis_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [base_pts], isClosed=False, color=(0, 0, 255), thickness=2)

    # --- SAVE XML ---
    tree = ET.ElementTree(pc_gts)
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="\t", level=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)

    # --- SAVE VISUALIZATION ---
    if save_vis and vis_img is not None:
        # Save alongside the XML with a _viz suffix
        vis_output_path = output_path.parent / f"{output_path.stem}_viz.jpg"
        cv2.imwrite(str(vis_output_path), vis_img)
        print(f"Visualization saved to: {vis_output_path}")


---
gnn_training/__init__.py
---

---
gnn_training/training/engine.py
---
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
---
gnn_training/training/__init__.py
---

---
gnn_training/training/main_train_eval.py
---
# training/main_train_eval.py
import yaml
import logging
from pathlib import Path
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import wandb
import argparse

# Local imports
# from ..gnn_data_preparation.config_models import DatasetCreationConfig
from ..gnn_data_preparation.dataset_generator import HistoricalLayoutGNNDataset
from ..gnn_data_preparation.utils import setup_logging
from .utils import set_seed, get_device, save_checkpoint, calculate_class_weights, create_prediction_log, FocalLoss
from .models.gnn_models import get_gnn_model
from .models.sklearn_models import get_sklearn_model
from .engine import train_one_epoch, evaluate
from .metrics import calculate_metrics
from .visualization import visualize_graph_predictions

logging.basicConfig(
    filename="train.log",       # file to save logs
    filemode="w",               # overwrite each run, use "a" to append
    level=logging.INFO,         # minimum level to log
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- EarlyStopping Class Definition ---
class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, mode='max',
                 checkpoint_path=None, save_checkpoint_fn=None, save_checkpoints_enabled=True):
        """
        Args:
            patience (int): How many epochs to wait after last validation metric improvement.
                            Default: 7
            verbose (bool): If True, prints a message for each validation metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
                            Default: 0
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity
                        monitored has stopped decreasing; in 'max' mode it will stop when the
                        quantity monitored has stopped increasing. Default: 'max'.
            checkpoint_path (Path): Path where to save the best model checkpoint. Required if save_checkpoint_fn
                                    is provided and save_checkpoints_enabled is True.
            save_checkpoint_fn (callable): The function to call to save the checkpoint.
                                          Expected signature: `func(model, optimizer, epoch, val_metrics, path)`.
            save_checkpoints_enabled (bool): If False, the early stopper will not save any checkpoints,
                                             but still track the best score and trigger early stopping.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.save_checkpoint_fn = save_checkpoint_fn
        self.save_checkpoints_enabled = save_checkpoints_enabled

        if self.mode == 'min':
            self.val_score_multiplier = -1
        elif self.mode == 'max':
            self.val_score_multiplier = 1
        else:
            raise ValueError(f"Mode {self.mode} not supported. Must be 'min' or 'max'.")

        if self.save_checkpoints_enabled and (not self.save_checkpoint_fn or not self.checkpoint_path):
            logging.warning("EarlyStopping initialized with save_checkpoints_enabled=True, but save_checkpoint_fn or checkpoint_path is missing. Checkpoints will NOT be saved.")
            self.save_checkpoints_enabled = False # Disable saving if arguments are incomplete

    def __call__(self, current_metric_value, model, optimizer, epoch, val_metrics_dict_for_save):
        """
        Call this method after each validation epoch.
        Args:
            current_metric_value (float): The current value of the monitored metric (e.g., val_f1_score).
            model: The model to save.
            optimizer: The optimizer to save.
            epoch: The current epoch number.
            val_metrics_dict_for_save (dict): Full dictionary of validation metrics to save in checkpoint.
                                              This will be passed to save_checkpoint_fn.
        """
        # Flip sign if 'max' mode to make it a minimization problem for consistent comparison
        score = self.val_score_multiplier * current_metric_value 
        logging.info(f'The precision is: {current_metric_value}')

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch, val_metrics_dict_for_save)
        elif score < self.best_score + self.delta: 
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # Improved
            if self.verbose:
                logging.info(f'Validation metric improved ({self.val_score_multiplier * self.best_score:.6f} --> {current_metric_value:.6f}). Saving model ...')
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch, val_metrics_dict_for_save)
            self.counter = 0

    def _save_checkpoint(self, model, optimizer, epoch, val_metrics_dict_for_save):
        """Saves model when validation metric improves, if saving is enabled."""
        if self.save_checkpoints_enabled:
            self.save_checkpoint_fn(model, optimizer, epoch, val_metrics_dict_for_save, self.checkpoint_path)
        elif self.verbose:
            logging.debug("Skipping model save as save_checkpoints is disabled in config.")


def run_gnn_fold(config, fold_dir, model_name, run_dir):
    """Trains and evaluates a GNN model for a single fold."""
    device = get_device(config['device'])
    
    # 1. Load Data
    train_dataset = HistoricalLayoutGNNDataset(root=str(fold_dir / 'gnn' / 'train'))
    val_dataset = HistoricalLayoutGNNDataset(root=str(fold_dir / 'gnn' / 'val'))
    test_dataset = HistoricalLayoutGNNDataset(root=str(fold_dir / 'gnn' / 'test'))

    
    train_loader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'], shuffle=True,num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training_params']['batch_size'], shuffle=False,num_workers=0, pin_memory=False)

    # 2. Initialize Model, Optimizer, Loss
    model = get_gnn_model(model_name, config, train_dataset[0]).to(device)
    optimizer = getattr(torch.optim, config['training_params']['optimizer'])(model.parameters(), lr=config['training_params']['learning_rate'])
    
    # --- START OF CHANGE: Learning Rate Warmup Scheduler ---
    warmup_epochs = config['training_params'].get('lr_warmup_epochs', 0)
    scheduler = None
    if warmup_epochs > 0:
        def warmup_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # Linearly increase the learning rate multiplier from a small fraction to 1
                return float(current_epoch + 1) / float(warmup_epochs)
            else:
                # After warmup, the multiplier is 1 (i.e., use the base_lr)
                return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        logging.info(f"Using learning rate warmup for {warmup_epochs} epochs.")
    # --- END OF CHANGE ---

    if config['training_params']['imbalance_handler'] == 'weighted_loss':
        weights = calculate_class_weights(train_dataset).to(device)
        logging.info(f"Using weighted cross-entropy with weights: {weights}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    elif config['training_params']['imbalance_handler'] == 'focal_loss':
        logging.info("Using Focal Loss")
        loss_fn = FocalLoss(alpha=config['training_params']['focal_loss_alpha'], gamma=config['training_params']['focal_loss_gamma'])
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        
    # --- Early Stopping Setup ---
    # Extract early stopping parameters from config, with sensible defaults
    early_stopping_patience = config['training_params'].get('early_stopping_patience', 10)
    early_stopping_min_delta = config['training_params'].get('early_stopping_min_delta', 0.0001)
    early_stopping_mode = config['training_params'].get('early_stopping_mode', 'max') # 'max' for metrics like F1, 'min' for loss

    # The metric to monitor for early stopping is the same as the checkpoint metric.
    # We need to extract the raw metric name from the config, e.g., 'val_f1_score' -> 'f1_score'.
    monitor_metric_key = config['checkpoint_metric'].replace('val_', '') 
    
    # Initialize EarlyStopping, passing the save_checkpoint function
    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        verbose=True, # Set to True to see messages about improvements and counters
        delta=early_stopping_min_delta,
        mode=early_stopping_mode,
        checkpoint_path=run_dir / 'best_model.pt',
        save_checkpoint_fn=save_checkpoint, # Pass the existing save_checkpoint utility
        save_checkpoints_enabled=config['save_checkpoints'] # Respect the global config for saving
    )

    logging.info("--- PRE-TRAINING LOADER TEST ---")
    try:
        # We will use the *exact same loader* that crashes later
        test_batch = next(iter(val_loader)) 
        logging.info(">>> SUCCESS: next(iter(val_loader)) worked perfectly before training. <<<")
        # It's a good idea to reset the iterator, though it will be re-created in the evaluate function
        val_loader_iterator = iter(val_loader)
    except Exception as e:
        logging.error(f"FATAL: val_loader crashed even BEFORE training. Error: {e}")

    # Now, start your normal training loop
    logging.info("--- Starting Training Loop ---")

    # 3. Training Loop
    for epoch in range(1, config['training_params']['epochs'] + 1):
        logging.info(f"--- Epoch {epoch}/{config['training_params']['epochs']} ---")
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch number {epoch}")
        val_metrics, _, _ = evaluate(model, val_loader, loss_fn, device)
        print("evaluation done")
        log_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        log_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        # --- START OF CHANGE: Log LR and Step Scheduler ---
        current_lr = optimizer.param_groups[0]['lr']
        log_metrics['learning_rate'] = current_lr
        
        logging.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Val Textline F1: {val_metrics['textline_f1_score']:.4f}, LR: {current_lr:.6f}")
        
        if scheduler:
            scheduler.step()
        # --- END OF CHANGE ---

        if config['tracking']['use_tracker'] == 'wandb':
            wandb.log(log_metrics, step=epoch)

        # --- Early Stopping Check and Checkpoint Saving ---
        current_metric_for_es = val_metrics[monitor_metric_key]
        early_stopper(current_metric_for_es, model, optimizer, epoch, val_metrics)

        if early_stopper.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch} due to no improvement in '{monitor_metric_key}' for {early_stopping_patience} epochs.")
            break # Exit the training loop
    
    # 4. Final Evaluation on Test Set
    logging.info("--- Final Evaluation on Test Set ---")
    checkpoint = torch.load(run_dir / 'best_model.pt', map_location=device)
    model = checkpoint['model'] # <-- LOAD THE MODEL OBJECT (As requested)

    # --- START OF CHANGE ---
    # Load original textline labels into a dictionary keyed by page_id
    # true_textline_labels_by_page = {}
    # data_creation_config_path = Path(config['dataset_path']).parent.parent /'configs/1_dataset_creation_config.yaml'
    # # This logic assumes the Phase 1 config is available to find the raw data dir
    # # A more robust way would be to save this info in the dataset itself
    # with open(data_creation_config_path, 'r') as f:
    #     d_config_dict = yaml.safe_load(f)
        
    # raw_data_dir = Path(d_config_dict['input_data_dir'])
    
    # for data in test_dataset:
    #     try:
    #         label_path = raw_data_dir / f"{data.page_id}_labels_textline.txt"
    #         true_textline_labels_by_page[data.page_id] = np.loadtxt(label_path, dtype=int)
    #     except Exception as e:
    #         logging.warning(f"Could not load textline labels for page {data.page_id}: {e}")
            
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, loss_fn, device)

    print("evaluate function run")

    log_test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    logging.info(f"Test Metrics: {log_test_metrics}")
    if config['tracking']['use_tracker'] == 'wandb':
        wandb.log(log_test_metrics)

    
    
    # 5. Save artifacts
    if config['save_prediction_log']:
        pred_df = create_prediction_log(test_dataset, test_preds, test_labels, model_name, fold_dir.name.split('_')[-1])
        pred_df.to_csv(run_dir / 'test_predictions.csv', index=False)

    if config['generate_visualizations']:
        pred_idx = 0
        for i, data in enumerate(test_dataset):
            print(f"visualizing {i}")
            if i >= config['num_visualizations']: break
            num_edges = data.edge_index.shape[1]
            page_preds = test_preds[pred_idx : pred_idx + num_edges]
            viz_path = run_dir / 'visualizations' / f"page_{data.page_id}.png"
            visualize_graph_predictions(data.cpu(), page_preds, viz_path)
            pred_idx += num_edges
            
    return log_test_metrics

def run_sklearn_fold(config, fold_dir, model_name, run_dir):
    """Trains and evaluates a scikit-learn model for a single fold."""
    # 1. Load Data
    train_df = pd.read_csv(fold_dir / 'sklearn' / 'train.csv')
    test_df = pd.read_csv(fold_dir / 'sklearn' / 'test.csv')

    X_train = train_df.drop(columns=['label', 'page_id', 'source_node_id', 'target_node_id'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label', 'page_id', 'source_node_id', 'target_node_id'])
    y_test = test_df['label']

    # 2. Train Model
    model = get_sklearn_model(model_name, config)
    logging.info(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # 3. Evaluate
    logging.info(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred) # Sklearn models don't have graph-level metrics here
    log_metrics = {f"test_{k}": v for k, v in metrics.items()}
    
    logging.info(f"Test Metrics for {model_name}: {log_metrics}")
    if config['tracking']['use_tracker'] == 'wandb':
         wandb.log(log_metrics)
         
    return log_metrics


def main():
    """Main execution function to run model training and cross-validation."""
    # --- NEW: Setup command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Run model training and cross-validation.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/2_training_config.yaml',
        help='Path to the YAML configuration file for training.'
    )
    parser.add_argument('--dataset_path', type=str, help='Override the dataset path from the config file.')
    parser.add_argument('--output_dir', type=str, help='Override the output directory from the config file.')
    parser.add_argument('--unique_folder_name', type=str, help='name of the unique folder specifiying datasize and grouping for this run.')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=None,
        help='Specific GPU index to use (e.g., 0, 1). If provided, overrides the device setting in the config file.'
    )
    args = parser.parse_args()

    # Load configuration from the path specified in the arguments
    config_path = Path(args.config)
    if not config_path.exists():
        logging.critical(f"Configuration file not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['random_seed'])

    # --- MODIFIED: Prioritize command-line arguments over config values ---
    dataset_path = Path(args.dataset_path) if args.dataset_path else Path(config['dataset_path'])
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output_dir'])
    
    # --- START OF CHANGE 2: Override config['device'] based on --gpu_id ---
    if args.gpu_id is not None:
        if args.gpu_id < torch.cuda.device_count():
            # Set the device string to 'cuda:X'
            config['device'] = f'cuda:{args.gpu_id}'
            logging.info(f"Overriding device to: {config['device']} from command line argument --gpu_id.")
        else:
             logging.warning(f"GPU ID {args.gpu_id} requested, but only {torch.cuda.device_count()} available. Using config default or falling back.")
             
    # Ensure a sensible default if the config itself is missing the device key (optional, but good practice)
    if 'device' not in config:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- END OF CHANGE 2 ---


    folds_dir = dataset_path / "folds"
    if not folds_dir.exists():
        logging.critical(f"Folds directory not found at {folds_dir}")
        return
        
    all_results = []

    for model_name in config['models_to_run']:
        model_type = config['model_configs'][model_name]['type']
        
        for fold_dir in sorted(folds_dir.glob('fold_*')):
            fold_idx = int(fold_dir.name.split('_')[-1])
            logging.info(f"===== Starting Fold {fold_idx} for Model: {model_name} =====")
            
            # Setup run directory and logging, now using the potentially overridden output_dir
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = config['tracking']['run_name_template'].format(model_name=model_name, fold_idx=fold_idx, timestamp=timestamp)
            run_dir = output_dir / args.unique_folder_name / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            setup_logging(run_dir / 'training.log')
            
            # Initialize tracker
            if config['tracking']['use_tracker'] == 'wandb':
                wandb.init(
                    project=config['project_name'],
                    name=run_name,
                    config=config,
                    reinit=True # Important for loops
                )
            
            if model_type == 'gnn':
                results = run_gnn_fold(config, fold_dir, model_name, run_dir)
            elif model_type == 'sklearn':
                results = run_sklearn_fold(config, fold_dir, model_name, run_dir)
            else:
                raise ValueError(f"Unknown model type {model_type}")
            
            results['model'] = model_name
            results['fold'] = fold_idx
            all_results.append(results)
            
            if config['tracking']['use_tracker'] == 'wandb':
                wandb.finish()

    # Aggregate and save final results, using the potentially overridden output_dir
    results_df = pd.DataFrame(all_results)
    agg_results = results_df.groupby('model').agg(['mean', 'std']).reset_index()
    
    logging.info("\n\n===== Aggregated Cross-Validation Results =====")
    print(agg_results)
    
    # Ensure the base output directory exists before saving summary files
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_results.to_csv(output_dir / 'aggregated_results.csv', index=False)
    results_df.to_csv(output_dir / 'all_fold_results.csv', index=False)
    
if __name__ == "__main__":
    main()
---
gnn_training/training/metrics.py
---
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

---
gnn_training/training/models/gnn_models.py
---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, SGConv, GATv2Conv, SplineConv

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron for edge prediction."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class EdgeClassifier(nn.Module):
    """
    A generic GNN-based edge classifier.
    It takes a GNN backbone and uses its output to predict edge labels.
    """
    def __init__(self, backbone, in_channels, edge_feat_dim, hidden_channels, dropout):
        super().__init__()
        self.backbone = backbone
        
        # The MLP takes the concatenated features of two nodes and optional edge features
        mlp_in_channels = 2 * hidden_channels + edge_feat_dim
        self.predictor = MLP(mlp_in_channels, hidden_channels, 2, num_layers=3, dropout=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # 1. Get node embeddings from the GNN backbone
        node_embeds = self.backbone(x, edge_index, edge_attr)

        # 2. For each edge, concatenate the embeddings of its source and target nodes
        source_embeds = node_embeds[edge_index[0]]
        target_embeds = node_embeds[edge_index[1]]
        
        edge_representation = torch.cat([source_embeds, target_embeds], dim=1)
        
        # 3. (Optional) Concatenate edge features
        if edge_attr is not None:
            edge_representation = torch.cat([edge_representation, edge_attr], dim=1)

        # 4. Predict edge logits
        edge_logits = self.predictor(edge_representation)
        return edge_logits

# --- GNN Backbone Definitions ---

class GCNBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None): # edge_attr is unused by GCN
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GATBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        # Edge attributes can be passed to GATv2Conv if needed, but it's not the standard implementation
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) -1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# --- START OF SGC MODIFICATION ---

class SGCBackbone(nn.Module):
    """
    A Simple Graph Convolution backbone.
    It consists of a single SGConv layer that propagates features K times.
    """
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        # The 'num_layers' from config corresponds to the 'K' parameter in SGConv
        self.conv = SGConv(in_channels, hidden_channels, K=num_layers, cached=False)

    def forward(self, x, edge_index, edge_attr=None):
        # SGConv does not use edge_attr or dropout between layers
        return self.conv(x, edge_index)

# --- END OF SGC MODIFICATION ---


# --- START OF MPNN MODIFICATION ---

class MPNNLayer(MessagePassing):
    """
    A single layer of the Message Passing Neural Network that uses edge attributes.
    """
    def __init__(self, in_channels, out_channels, edge_dim, dropout):
        super().__init__(aggr='add') # "Add" aggregation.
        # This network computes the message based on source, target, and edge features.
        # Message = MLP(source_node || target_node || edge_feature)
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # This network updates the node features after aggregation.
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_dim]

        # propagate() will call message() and update()
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update node features: MLP(old_node_features || aggregated_messages)
        return self.update_net(torch.cat([x, out], dim=1))

    def message(self, x_i, x_j, edge_attr):
        # x_i: Source node features [E, in_channels]
        # x_j: Target node features [E, in_channels]
        # edge_attr: Edge features [E, edge_dim]
        
        # Concatenate features and pass through the message network
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(tmp)


class MPNNBackbone(nn.Module):
    """
    A multi-layer MPNN backbone that uses edge attributes.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, edge_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer maps from input channels to hidden channels
        self.layers.append(MPNNLayer(in_channels, hidden_channels, edge_dim, dropout))
        # Subsequent layers map from hidden channels to hidden channels
        for _ in range(num_layers - 1):
            self.layers.append(MPNNLayer(hidden_channels, hidden_channels, edge_dim, dropout))

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            # Create a dummy tensor if edge attributes are missing
            edge_attr = torch.zeros((edge_index.shape[1], 1), device=x.device)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class SplineConvBackbone(nn.Module):
    """
    A multi-layer SplineConv backbone that uses edge attributes.
    It includes a pre-processing step to reduce high-dimensional edge features.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, edge_dim, kernel_size, spline_edge_dim=3):
        super().__init__()
        
        self.edge_dim = edge_dim 
        self.spline_edge_dim = spline_edge_dim
        
        if self.edge_dim == 0:
            self.edge_preprocessor = None
        else:
            # self.edge_preprocessor = nn.Linear(self.edge_dim, self.spline_edge_dim)
            # --- MODIFICATION START: Using an MLP instead of a single Linear layer ---
            # Define an MLP with one hidden layer (e.g., of size 2 * spline_edge_dim)
            mlp_hidden_dim = 2 * spline_edge_dim 
            self.edge_preprocessor = nn.Sequential(
                nn.Linear(self.edge_dim, mlp_hidden_dim),
                nn.ReLU(),  # Add a non-linear activation function
                # nn.Dropout(p=0.2), # Optional: Add dropout for regularization
                nn.Linear(mlp_hidden_dim, self.spline_edge_dim)
            )
            # --- MODIFICATION END ---

        self.convs = nn.ModuleList()
        self.convs.append(SplineConv(in_channels, hidden_channels, dim=self.spline_edge_dim, kernel_size=kernel_size, aggr='sum'))
        for _ in range(num_layers - 1):
            self.convs.append(SplineConv(hidden_channels, hidden_channels, dim=self.spline_edge_dim, kernel_size=kernel_size, aggr='sum'))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            processed_edge_attr = torch.zeros((edge_index.shape[1], self.spline_edge_dim), device=x.device)
        else:
            processed_edge_attr = self.edge_preprocessor(edge_attr)
            
            processed_edge_attr = torch.sigmoid(processed_edge_attr)
  
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, processed_edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# --- Model Factory ---
def get_gnn_model(model_name: str, config: dict, data_sample) -> nn.Module:
    """Factory function to create a GNN model for training."""
    model_cfg = config['model_configs'][model_name]
    
    in_channels = data_sample.num_node_features
    # This is the TRUE edge feature dimension from the data (can be 0)
    edge_feat_dim = data_sample.num_edge_features if hasattr(data_sample, 'edge_attr') and data_sample.edge_attr is not None else 0
    
    if in_channels == 0:
        raise ValueError("Input data has 0 node features.")
        
    if model_name == "GCN":
        backbone = GCNBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'], model_cfg['dropout'])
    elif model_name == "GAT":
        backbone = GATBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'], model_cfg['heads'], model_cfg['dropout'])
    elif model_name == "SGC":
        backbone = SGCBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'])
    
    # --- SIMPLIFICATION STARTS HERE ---
    elif model_name == "MPNN":
        # Let MPNN handle the case where edge_feat_dim might be 0
        backbone = MPNNBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'], model_cfg['dropout'], edge_dim=edge_feat_dim)
    elif model_name == "SplineCNN":
        # Pass the true edge_feat_dim. The backbone will handle the reduction.
        # Also pass the new spline_edge_dim from the config if it exists.
        spline_dim = model_cfg.get('spline_edge_dim', 3) # Default to 3 if not in config
        backbone = SplineConvBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'], model_cfg['dropout'], edge_dim=edge_feat_dim, kernel_size=model_cfg['kernel_size'], spline_edge_dim=spline_dim)
    # --- SIMPLIFICATION ENDS HERE ---
    
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")

    # Determine the number of channels output by the backbone for the final predictor MLP
    if model_name == "GAT":
        final_hidden_channels = model_cfg['hidden_channels'] * model_cfg['heads']
    else:
        final_hidden_channels = model_cfg['hidden_channels']

    # For the predictor, use the original edge_feat_dim, as it concatenates the original features
    return EdgeClassifier(backbone, in_channels, edge_feat_dim, final_hidden_channels, model_cfg['dropout'])
---
gnn_training/training/models/__init__.py
---

---
gnn_training/training/models/sklearn_models.py
---
# training/models/sklearn_models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

def get_sklearn_model(model_name: str, config: dict) -> BaseEstimator:
    """Factory function to create a scikit-learn model."""
    model_cfg = config['model_configs'][model_name]
    
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=model_cfg['n_estimators'],
            max_depth=model_cfg['max_depth'],
            random_state=config['random_seed'],
            n_jobs=model_cfg['n_jobs']
        )
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=model_cfg['n_estimators'],
            learning_rate=model_cfg['learning_rate'],
            max_depth=model_cfg['max_depth'],
            random_state=config['random_seed']
        )
    elif model_name == "LogisticRegression":
        return LogisticRegression(
            C=model_cfg['C'],
            max_iter=model_cfg['max_iter'],
            random_state=config['random_seed'],
            n_jobs=model_cfg.get('n_jobs', -1)
        )
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")
---
gnn_training/training/utils.py
---
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
---
gnn_training/training/visualization.py
---
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
---
inference.py
---
import os
import argparse
import gc
from PIL import Image
import torch

from segmentation.segment_graph import images2points




def process_new_manuscript(manuscript_path, target_longest_side=2500):
    source_images_path = os.path.join(manuscript_path, "images")
    # We will save processed (and potentially resized) images here
    # to avoid modifying source files while iterating over them.
    resized_images_path = os.path.join(manuscript_path, "images_resized")

    try:
        # Create the target folder
        os.makedirs(resized_images_path, exist_ok=True)
        
        # Verify source exists
        if not os.path.exists(source_images_path):
            print(f"Error: Source directory {source_images_path} not found.")
            return

    except Exception as e:
        print(f"An error occurred setting up directories: {e}")
        return

    # Valid image extensions to look for
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    # Get list of files in the directory
    files = [f for f in os.listdir(source_images_path) if os.path.isfile(os.path.join(source_images_path, f))]

    print(f"Found {len(files)} files in {source_images_path}...")

    for filename in files:
        # Skip non-image files based on extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        base_filename = os.path.splitext(filename)[0]
        file_path = os.path.join(source_images_path, filename)

        try:
            # Open the image from the folder
            with Image.open(file_path) as image:
                
                width, height = image.size
                
                # 1. VALIDATION: Check if image is too small for CV tasks
                # If both dimensions are smaller than 600, we reject the image.
                if width < 600 and height < 600:
                    raise ValueError(f"Image resolution too low ({width}x{height}). Both dimensions are < 600px.")

                
                # Check if the longest side exceeds the target
                if max(width, height) > target_longest_side:
                    
                    # Calculate scaling factor
                    scale_factor = target_longest_side / max(width, height)
                    
                    # Calculate new dimensions
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # Handle Resampling filter compatibility
                    try:
                        resampling_filter = Image.Resampling.LANCZOS
                    except AttributeError:
                        resampling_filter = Image.LANCZOS

                    print(f"Downscaling '{filename}': ({width}x{height}) -> ({new_width}x{new_height})")
                    image = image.resize((new_width, new_height), resampling_filter)
                    
                else:
                    print(f"Image '{filename}' is within limits ({width}x{height}). Keeping original size.")
                    

                # Standardize Color Mode
                if image.mode in ("RGBA", "P", "LA"):
                    image = image.convert("RGB")

                # Save processed image to the NEW folder
                new_filename = f"{base_filename}.jpg"
                save_path = os.path.join(resized_images_path, new_filename)
                
                image.save(save_path, "JPEG")
                print(f"Processed: {new_filename}")

        except Exception as img_err:
            # This block catches the ValueError raised above and prints the message
            print(f"Failed to process image {filename}: {img_err}")
            continue

    # Point the inference function to the new resized/processed folder
    print("Running images2points on processed folder...")
    images2points(resized_images_path) 
    
    # Cleanup resources
    torch.cuda.empty_cache()
    gc.collect()

    print("Processing complete.")






# if __name__ == "__main__":
#     # 1. Parse standard CLI arguments4
#     parser = argparse.ArgumentParser(description="GNN Layout Analysis Inference")
#     parser.add_argument("--manuscript_path", type=str, default="./input_manuscripts/sample_manuscript_1", help="Path to the manuscript directory")
#     args = parser.parse_args()

#     # the data preparation.yaml is tied to the model_checkpoint used.
#     args.model_checkpoint = "./pretrained_gnn/v2.pt"
#     args.dataset_config_path = "./pretrained_gnn/gnn_preprocessing_v2.yaml"

#     # -- Hyperparameters
#     args.visualize = True
#     args.BINARIZE_THRESHOLD = 0.5098
#     args.BBOX_PAD_V = 0.7
#     args.BBOX_PAD_H = 0.5
#     args.CC_SIZE_THRESHOLD_RATIO = 0.4

#     process_new_manuscript(args.manuscript_path)
#     run_gnn_inference(args)




---
my-app/.env
---
VITE_BACKEND_URL="http://localhost:5000"
---
my-app/env.d.ts
---
/// <reference types="vite/client" />

---
my-app/.gitignore
---
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

node_modules
.DS_Store
dist
dist-ssr
coverage
*.local

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?

*.tsbuildinfo

.eslintcache

# Cypress
/cypress/videos/
/cypress/screenshots/

# Vitest
__screenshots__/

---
my-app/index.html
---
<!DOCTYPE html>
<html lang="">
  <head>
    <meta charset="UTF-8">
    <link rel="icon" href="/favicon.ico">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vite App</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>

---
my-app/package.json
---
{
  "name": "my-app",
  "version": "0.0.0",
  "private": true,
  "type": "module",
  "engines": {
    "node": "^20.19.0 || >=22.12.0"
  },
  "scripts": {
    "dev": "vite",
    "build": "run-p type-check \"build-only {@}\" --",
    "preview": "vite preview",
    "build-only": "vite build",
    "type-check": "vue-tsc --build"
  },
  "dependencies": {
    "pinia": "^3.0.4",
    "vue": "^3.5.26",
    "vue-router": "^4.6.4"
  },
  "devDependencies": {
    "@tsconfig/node24": "^24.0.3",
    "@types/node": "^24.10.4",
    "@vitejs/plugin-vue": "^6.0.3",
    "@vue/tsconfig": "^0.8.1",
    "npm-run-all2": "^8.0.4",
    "typescript": "~5.9.3",
    "vite": "^7.3.0",
    "vite-plugin-vue-devtools": "^8.0.5",
    "vue-tsc": "^3.2.2"
  }
}

---
my-app/README.md
---
# my-app

This template should help get you started developing with Vue 3 in Vite.

## Recommended IDE Setup

[VS Code](https://code.visualstudio.com/) + [Vue (Official)](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur).

## Recommended Browser Setup

- Chromium-based browsers (Chrome, Edge, Brave, etc.):
  - [Vue.js devtools](https://chromewebstore.google.com/detail/vuejs-devtools/nhdogjmejiglipccpnnnanhbledajbpd)
  - [Turn on Custom Object Formatter in Chrome DevTools](http://bit.ly/object-formatters)
- Firefox:
  - [Vue.js devtools](https://addons.mozilla.org/en-US/firefox/addon/vue-js-devtools/)
  - [Turn on Custom Object Formatter in Firefox DevTools](https://fxdx.dev/firefox-devtools-custom-object-formatters/)

## Type Support for `.vue` Imports in TS

TypeScript cannot handle type information for `.vue` imports by default, so we replace the `tsc` CLI with `vue-tsc` for type checking. In editors, we need [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) to make the TypeScript language service aware of `.vue` types.

## Customize configuration

See [Vite Configuration Reference](https://vite.dev/config/).

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Type-Check, Compile and Minify for Production

```sh
npm run build
```

---
my-app/src/App.vue
---
<template>
  <div class="app-container">
    <div v-if="!currentManuscript">
      <!-- Upload Screen -->
      <div class="upload-card">
        <h1>Historical Manuscript Segmentation</h1>
        <div class="form-group">
          <label>Manuscript Name:</label>
          <input v-model="formName" type="text" placeholder="e.g. manuscript_1" />
        </div>
        <div class="form-group">
          <label>Resize Dimension (Longest Side):</label>
          <input v-model.number="formLongestSide" type="number" />
        </div>
        <div class="form-group">
          <label>Images:</label>
          <input type="file" multiple @change="handleFileChange" accept="image/*" />
        </div>
        <button @click="upload" :disabled="uploading">
          {{ uploading ? 'Processing (Step 1-3)...' : 'Start Processing' }}
        </button>
        <div v-if="uploadStatus" class="status">{{ uploadStatus }}</div>
      </div>
    </div>

    <div v-else>
      <!-- Main Workstation -->
      <button class="back-btn" @click="currentManuscript = null">← Back to Upload</button>
      <ManuscriptViewer 
        :manuscriptName="currentManuscript" 
        :pageName="currentPage"
        @page-changed="handlePageChange"
      />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ManuscriptViewer from './components/ManuscriptViewer.vue'

// Basic State
const currentManuscript = ref(null)
const currentPage = ref(null)
const pageList = ref([])

// Upload Form State
const formName = ref('my_manuscript')
const formLongestSide = ref(2500)
const selectedFiles = ref([])
const uploading = ref(false)
const uploadStatus = ref('')

const handleFileChange = (e) => {
  selectedFiles.value = Array.from(e.target.files)
}

const upload = async () => {
  if (selectedFiles.value.length === 0) return alert('Select files')
  uploading.value = true
  uploadStatus.value = 'Uploading and generating heatmaps/points. This may take a while...'

  const formData = new FormData()
  formData.append('manuscriptName', formName.value)
  formData.append('longestSide', formLongestSide.value)
  selectedFiles.value.forEach(file => formData.append('images', file))

  try {
    const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/upload`, {
      method: 'POST',
      body: formData
    })
    if(!res.ok) throw new Error('Upload failed')
    const data = await res.json()
    
    // Success - Switch to viewer
    pageList.value = data.pages
    if (pageList.value.length > 0) {
      currentManuscript.value = formName.value
      currentPage.value = pageList.value[0]
    } else {
      uploadStatus.value = 'No pages processed.'
    }
  } catch (e) {
    uploadStatus.value = 'Error: ' + e.message
  } finally {
    uploading.value = false
  }
}

const handlePageChange = (newPage) => {
  currentPage.value = newPage
}
</script>

<style>
body { margin: 0; font-family: sans-serif; background: #222; color: white; }
.app-container { display: flex; flex-direction: column; height: 100vh; }
.upload-card { max-width: 500px; margin: 100px auto; padding: 20px; background: #333; border-radius: 8px; }
.form-group { margin-bottom: 15px; display: flex; flex-direction: column; }
input { padding: 8px; background: #444; border: 1px solid #555; color: white; margin-top: 5px; }
button { padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }
button:disabled { background: #555; }
.back-btn { position: absolute; top: 10px; left: 10px; z-index: 1000; background: #444; }
</style>
---
my-app/src/components/ManuscriptViewer.vue
---
<template>
  <div class="manuscript-viewer">
    <!-- Top Toolbar: Collapsible -->
    <div class="toolbar">
      <h10>{{ manuscriptNameForDisplay }} - Page {{ currentPageForDisplay }}</h10>
      <div v-show="!isToolbarCollapsed" class="toolbar-controls">
        <button @click="previousPage" :disabled="loading || isProcessingSave || isFirstPage">
          Previous
        </button>
        <button @click="nextPage" :disabled="loading || isProcessingSave || isLastPage">Next</button>
        <button @click="saveAndGoNext" :disabled="loading || isProcessingSave">
          Save & Next (S)
        </button>
        <button @click="goToIMG2TXTPage" :disabled="loading || isProcessingSave">
          Annotate Text
        </button>
        <div class="toggle-container">
          <label>
            <input type="checkbox" v-model="textlineModeActive" :disabled="isProcessingSave" />
            Edge Edit (W)
          </label>
        </div>
        <div class="toggle-container">
          <label>
            <input
              type="checkbox"
              v-model="textboxModeActive"
              :disabled="isProcessingSave || !graphIsLoaded"
            />
            Region Labeling (R)
          </label>
        </div>
      </div>
      <button @click="runHeuristic" :disabled="loading">Auto-Link (Heuristic)</button>
      <button class="panel-toggle-btn" @click="isToolbarCollapsed = !isToolbarCollapsed">
        {{ isToolbarCollapsed ? 'Show Toolbar' : 'Hide' }}
      </button>
    </div>

    <!-- Main Content: Visualization Area -->
    <div class="visualization-container" ref="container">
      <div v-if="isProcessingSave" class="processing-save-notice">
        Saving graph and processing... Please wait.
      </div>
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
      <div v-if="loading" class="loading">Loading Page Data...</div>
      <div
        v-else
        class="image-container"
        :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }"
      >
        <img
          v-if="imageData"
          :src="`data:image/jpeg;base64,${imageData}`"
          :width="scaledWidth"
          :height="scaledHeight"
          class="manuscript-image"
          @load="imageLoaded = true"
        />
        <div
          v-else
          class="placeholder-image"
          :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }"
        >
          No image available
        </div>

        <svg
          v-if="graphIsLoaded"
          class="graph-overlay"
          :class="{ 'is-visible': textlineModeActive || textboxModeActive }"
          :width="scaledWidth"
          :height="scaledHeight"
          :style="{ cursor: svgCursor }"
          @click="textlineModeActive && onBackgroundClick($event)"
          @mousemove="handleSvgMouseMove"
          @mouseleave="handleSvgMouseLeave"
          ref="svgOverlayRef"
        >
          <line
            v-for="(edge, index) in workingGraph.edges"
            :key="`edge-${index}`"
            :x1="scaleX(workingGraph.nodes[edge.source].x)"
            :y1="scaleY(workingGraph.nodes[edge.source].y)"
            :x2="scaleX(workingGraph.nodes[edge.target].x)"
            :y2="scaleY(workingGraph.nodes[edge.target].y)"
            :stroke="getEdgeColor(edge)"
            :stroke-width="isEdgeSelected(edge) ? 3 : 2.5"
            @click.stop="textlineModeActive && onEdgeClick(edge, $event)"
          />

          <circle
            v-for="(node, nodeIndex) in workingGraph.nodes"
            :key="`node-${nodeIndex}`"
            :cx="scaleX(node.x)"
            :cy="scaleY(node.y)"
            :r="getNodeRadius(nodeIndex)"
            :fill="getNodeColor(nodeIndex)"
            @click.stop="textlineModeActive && onNodeClick(nodeIndex, $event)"
          />

          <line
            v-if="
              textlineModeActive &&
              selectedNodes.length === 1 &&
              tempEndPoint &&
              !isAKeyPressed &&
              !isDKeyPressed
            "
            :x1="scaleX(workingGraph.nodes[selectedNodes[0]].x)"
            :y1="scaleY(workingGraph.nodes[selectedNodes[0]].y)"
            :x2="tempEndPoint.x"
            :y2="tempEndPoint.y"
            stroke="#ff9500"
            stroke-width="2.5"
            stroke-dasharray="5,5"
          />
        </svg>
      </div>
    </div>

    <!-- Bottom Panel: Collapsible -->
    <div class="bottom-panel">
      <div class="panel-toggle-bar" @click="isControlsCollapsed = !isControlsCollapsed">
        <div class="edit-instructions">
          <p v-if="isControlsCollapsed && textboxModeActive">
            Hold 'e' and hover over lines to label them. Release 'e' and press again for the next
            label. 's' to save.
          </p>
          <p v-else-if="isControlsCollapsed && textlineModeActive">
            Hold 'a' to connect, 'd' to delete. Press 's' to save & next. Toggle modes with 'w'/'r'.
          </p>
          <p v-else-if="isControlsCollapsed && !textlineModeActive && !textboxModeActive">
            Press 'w' to edit edges, 'r' to label regions.
          </p>
          <p v-else-if="textboxModeActive">
            Hold 'e' to label textlines with the current label. Release and press 'e' again to move
            to the next label.
          </p>
          <p v-else-if="textlineModeActive && !isAKeyPressed && !isDKeyPressed">
            Select nodes to manage edges, or use hotkeys.
          </p>
          <p v-else-if="textlineModeActive && isAKeyPressed">Release 'A' to connect nodes.</p>
          <p v-else-if="textlineModeActive && isDKeyPressed">Release 'D' to stop deleting.</p>
        </div>
        <button class="panel-toggle-btn">
          {{ isControlsCollapsed ? 'Show Controls' : 'Hide Controls' }}
        </button>
      </div>

      <div v-show="!isControlsCollapsed" class="bottom-panel-content">
        <div v-if="textlineModeActive && !isAKeyPressed && !isDKeyPressed" class="edit-controls">
          <div class="edit-actions">
            <button @click="resetSelection">Cancel Selection</button>
            <button
              @click="addEdge"
              :disabled="selectedNodes.length !== 2 || edgeExists(selectedNodes[0], selectedNodes[1])"
            >
              Add Edge
            </button>
            <button
              @click="deleteEdge"
              :disabled="selectedNodes.length !== 2 || !edgeExists(selectedNodes[0], selectedNodes[1])"
            >
              Delete Edge
            </button>
          </div>
        </div>

        <div
          v-if="(textlineModeActive || textboxModeActive) && graphIsLoaded"
          class="modifications-log-container"
        >
          <button @click="saveCurrentGraph" :disabled="loading || isProcessingSave">
            Save Graph & Labels
          </button>
          <div v-if="modifications.length > 0" class="modifications-details">
            <h3>Modifications ({{ modifications.length }})</h3>
            <button @click="resetModifications" :disabled="loading">Reset All Changes</button>
            <ul>
              <li
                v-for="(mod, index) in modifications"
                :key="index"
                class="modification-item"
              >
                {{ mod.type === 'add' ? 'Added' : 'Removed' }} edge: {{ mod.source }} ↔
                {{ mod.target }}
                <button @click="undoModification(index)" class="undo-button">Undo</button>
              </li>
            </ul>
          </div>
          <p v-else-if="!loading">No edge modifications in this session.</p>
        </div>
      </div>
    </div>
  </div>
</template>


<script setup>
import { ref, onMounted, onBeforeUnmount, computed, watch, reactive } from 'vue'
import { generateLayoutGraph } from '../layout-analysis-utils/LayoutGraphGenerator.js'
import { useRouter } from 'vue-router'

const props = defineProps({
  manuscriptName: {
    type: String,
    default: null,
  },
  pageName: {
    type: String,
    default: null,
  },
})

// Define emits so we can tell the parent App when to change the page
const emit = defineEmits(['page-changed'])

const router = useRouter()

// We determine if we are in specific router flow or the standalone app flow based on props
const isEditModeFlow = computed(() => !!props.manuscriptName && !!props.pageName)

const localManuscriptName = ref('')
const localCurrentPage = ref('')
const localPageList = ref([])

const loading = ref(true)
const isProcessingSave = ref(false)
const error = ref(null)
const imageData = ref('')
const imageLoaded = ref(false)

const isToolbarCollapsed = ref(true)
const isControlsCollapsed = ref(true)
const textlineModeActive = ref(false)
const textboxModeActive = ref(false)

const dimensions = ref([0, 0])
const points = ref([])
const graph = ref({ nodes: [], edges: [] })
const workingGraph = reactive({ nodes: [], edges: [] })
const modifications = ref([])
const nodeEdgeCounts = ref({})
const selectedNodes = ref([])
const tempEndPoint = ref(null)
const isDKeyPressed = ref(false)
const isAKeyPressed = ref(false)
const isEKeyPressed = ref(false) // For holding E
const hoveredNodesForMST = reactive(new Set())
const container = ref(null)
const svgOverlayRef = ref(null)

// --- State for region labeling ---
const textlineLabels = reactive({}) // Maps node index to a region label (0, 1, 2...)
const textlines = ref({}) // Maps textline ID to a list of node indices
const nodeToTextlineMap = ref({}) // Maps node index to its textline ID
const hoveredTextlineId = ref(null)
const textboxLabels = ref(0) // The current label to apply (0, 1, 2, ...)
const labelColors = ['#448aff', '#ffeb3b', '#4CAF50', '#f44336', '#9c27b0', '#ff9800'] // Colors for different labels

const scaleFactor = 0.7
const NODE_HOVER_RADIUS = 7
const EDGE_HOVER_THRESHOLD = 5

const manuscriptNameForDisplay = computed(() => localManuscriptName.value)
const currentPageForDisplay = computed(() => localCurrentPage.value)
const isFirstPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === 0)
const isLastPage = computed(
  () => localPageList.value.indexOf(localCurrentPage.value) === localPageList.value.length - 1
)

const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor))
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor))
const scaleX = (x) => x * scaleFactor
const scaleY = (y) => y * scaleFactor
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0)

const svgCursor = computed(() => {
  if (textboxModeActive.value) {
    if (isEKeyPressed.value) return 'crosshair'
    return 'pointer'
  }
  if (!textlineModeActive.value) return 'default'
  if (isAKeyPressed.value) return 'crosshair'
  if (isDKeyPressed.value) return 'not-allowed'
  return 'default'
})

const computeTextlines = () => {
  if (!graphIsLoaded.value) {
    // Safety: If graph isn't loaded, clear lines so we don't show stale data
    textlines.value = {}
    nodeToTextlineMap.value = {}
    return
  }

  const numNodes = workingGraph.nodes.length
  const adj = Array(numNodes)
    .fill(0)
    .map(() => [])

  // FIX 1: Add bounds checking to prevent crashes on bad data
  for (const edge of workingGraph.edges) {
    // Only add the edge if both source and target exist in our node list
    if (adj[edge.source] && adj[edge.target]) {
      adj[edge.source].push(edge.target)
      adj[edge.target].push(edge.source)
    }
  }

  const visited = new Array(numNodes).fill(false)
  const newTextlines = {}
  const newNodeToTextlineMap = {}
  let currentTextlineId = 0

  for (let i = 0; i < numNodes; i++) {
    if (!visited[i]) {
      const component = []
      const stack = [i]
      visited[i] = true
      while (stack.length > 0) {
        const u = stack.pop()
        component.push(u)
        newNodeToTextlineMap[u] = currentTextlineId
        for (const v of adj[u]) {
          if (!visited[v]) {
            visited[v] = true
            stack.push(v)
          }
        }
      }
      newTextlines[currentTextlineId] = component
      currentTextlineId++
    }
  }

  textlines.value = newTextlines
  nodeToTextlineMap.value = newNodeToTextlineMap
}

const fetchPageData = async (manuscript, page) => {
  if (!manuscript || !page) {
    error.value = 'Manuscript or page not specified.'
    loading.value = false
    return
  }
  loading.value = true
  error.value = null
  modifications.value = []
  Object.keys(textlineLabels).forEach((key) => delete textlineLabels[key])

  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscript}/${page}`
    )
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data')
    const data = await response.json()

    dimensions.value = data.dimensions
    imageData.value = data.image || ''
    points.value = data.points.map((p) => ({ coordinates: [p[0], p[1]], segment: null }))

    if (data.graph) {
      graph.value = data.graph
    } else if (data.points?.length > 0) {
      graph.value = generateLayoutGraph(data.points)
      if (!isEditModeFlow.value) {
        await saveGeneratedGraph(manuscript, page, graph.value)
      }
    }
    if (data.textline_labels) {
      data.textline_labels.forEach((label, index) => {
        if (label !== -1) {
          textlineLabels[index] = label
        }
      })
    }

    resetWorkingGraph()
  } catch (err) {
    console.error('Error fetching page data:', err)
    error.value = err.message
  } finally {
    loading.value = false
  }
}

const fetchPageList = async (manuscript) => {
  if (!manuscript) return
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/manuscript/${manuscript}/pages`
    )
    if (!response.ok) throw new Error('Failed to fetch page list')
    localPageList.value = await response.json()
  } catch (err) {
    console.error('Failed to fetch page list:', err)
    localPageList.value = []
  }
}

const updateUniqueNodeEdgeCounts = () => {
  const counts = {}
  if (!workingGraph.nodes) return
  workingGraph.nodes.forEach((_, index) => {
    counts[index] = 0
  })

  if (!workingGraph.edges) {
    nodeEdgeCounts.value = counts
    return
  }

  const uniqueEdges = new Set()
  for (const edge of workingGraph.edges) {
    const key = `${Math.min(edge.source, edge.target)}-${Math.max(edge.source, edge.target)}`
    uniqueEdges.add(key)
  }

  for (const key of uniqueEdges) {
    const [source, target] = key.split('-').map(Number)
    if (counts[source] !== undefined) counts[source]++
    if (counts[target] !== undefined) counts[target]++
  }

  nodeEdgeCounts.value = counts
}

watch(
  [() => workingGraph.edges, () => workingGraph.nodes],
  () => {
    updateUniqueNodeEdgeCounts()
    computeTextlines()
  },
  { deep: true, immediate: true }
)

const resetWorkingGraph = () => {
  workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []))
  workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []))
  resetSelection()
  computeTextlines()
}

const getNodeColor = (nodeIndex) => {
  const textlineId = nodeToTextlineMap.value[nodeIndex]

  if (textboxModeActive.value) {
    if (hoveredTextlineId.value !== null && hoveredTextlineId.value === textlineId) {
      return '#ff4081' // Hot pink for hovered textline
    }
    const label = textlineLabels[nodeIndex]
    if (label !== undefined && label > -1) {
      return labelColors[label % labelColors.length]
    }
    return '#9e9e9e' // Grey for unlabeled nodes in this mode
  }

  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return '#00bcd4'
  if (isNodeSelected(nodeIndex)) return '#ff9500'

  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (edgeCount < 2) return '#f44336'
  if (edgeCount === 2) return '#4CAF50'
  if (edgeCount > 2) return '#2196F3'
  return '#cccccc'
}
const getNodeRadius = (nodeIndex) => {
  if (textboxModeActive.value) {
    const textlineId = nodeToTextlineMap.value[nodeIndex]
    if (hoveredTextlineId.value !== null && hoveredTextlineId.value === textlineId) {
      return 7
    }
    return 5
  }

  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return 7
  if (isNodeSelected(nodeIndex)) return 6
  return edgeCount < 2 ? 5 : 3
}
const getEdgeColor = (edge) => (edge.modified ? '#f44336' : '#ffffff')
const isNodeSelected = (nodeIndex) => selectedNodes.value.includes(nodeIndex)
const isEdgeSelected = (edge) => {
  return (
    selectedNodes.value.length === 2 &&
    ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
      (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source))
  )
}

const resetSelection = () => {
  selectedNodes.value = []
  tempEndPoint.value = null
}
const onNodeClick = (nodeIndex, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value || textboxModeActive.value) return
  event.stopPropagation()
  const existingIndex = selectedNodes.value.indexOf(nodeIndex)
  if (existingIndex !== -1) selectedNodes.value.splice(existingIndex, 1)
  else
    selectedNodes.value.length < 2
      ? selectedNodes.value.push(nodeIndex)
      : (selectedNodes.value = [nodeIndex])
}
const onEdgeClick = (edge, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value || textboxModeActive.value) return
  event.stopPropagation()
  selectedNodes.value = [edge.source, edge.target]
}
const onBackgroundClick = () => {
  if (!isAKeyPressed.value && !isDKeyPressed.value) resetSelection()
}

const handleSvgMouseMove = (event) => {
  if (!svgOverlayRef.value) return
  const { left, top } = svgOverlayRef.value.getBoundingClientRect()
  const mouseX = event.clientX - left
  const mouseY = event.clientY - top

  if (textboxModeActive.value) {
    let newHoveredTextlineId = null

    // 1. Check for node hover first (more precise)
    for (let i = 0; i < workingGraph.nodes.length; i++) {
      const node = workingGraph.nodes[i]
      if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS) {
        newHoveredTextlineId = nodeToTextlineMap.value[i]
        break // Exit loop once found
      }
    }

    // 2. If no node hovered, check for edge hover
    if (newHoveredTextlineId === null) {
      for (const edge of workingGraph.edges) {
        const n1 = workingGraph.nodes[edge.source]
        const n2 = workingGraph.nodes[edge.target]
        if (
          n1 &&
          n2 &&
          distanceToLineSegment(
            mouseX,
            mouseY,
            scaleX(n1.x),
            scaleY(n1.y),
            scaleX(n2.x),
            scaleY(n2.y)
          ) < EDGE_HOVER_THRESHOLD
        ) {
          // An edge connects two nodes of the same textline, so we can use either.
          newHoveredTextlineId = nodeToTextlineMap.value[edge.source]
          break // Exit loop once found
        }
      }
    }

    // 3. Update the hovered textline ID
    hoveredTextlineId.value = newHoveredTextlineId

    // 4. Apply label if key is pressed
    if (hoveredTextlineId.value !== null && isEKeyPressed.value) {
      labelTextline()
    }
    return
  }

  if (!textlineModeActive.value) return
  if (isDKeyPressed.value) handleEdgeHoverDelete(mouseX, mouseY)
  else if (isAKeyPressed.value) handleNodeHoverCollect(mouseX, mouseY)
  else if (selectedNodes.value.length === 1) tempEndPoint.value = { x: mouseX, y: mouseY }
  else tempEndPoint.value = null
}

const handleSvgMouseLeave = () => {
  if (selectedNodes.value.length === 1) tempEndPoint.value = null
  hoveredTextlineId.value = null
}

const labelTextline = () => {
  if (hoveredTextlineId.value === null) return
  const nodesToLabel = textlines.value[hoveredTextlineId.value]
  if (nodesToLabel) {
    nodesToLabel.forEach((nodeIndex) => {
      textlineLabels[nodeIndex] = textboxLabels.value
    })
  }
}

const handleGlobalKeyDown = (e) => {
  const key = e.key.toLowerCase()

  // General hotkeys that work in multiple modes
  if (key === 's' && !e.repeat) {
    if (
      (textlineModeActive.value || textboxModeActive.value) &&
      !loading.value &&
      !isProcessingSave.value
    ) {
      e.preventDefault()
      saveAndGoNext()
    }
    return
  }
  if (key === 'w' && !e.repeat) {
    e.preventDefault()
    textlineModeActive.value = !textlineModeActive.value
    return
  }
  if (key === 'r' && !e.repeat) {
    e.preventDefault()
    textboxModeActive.value = !textboxModeActive.value
    return
  }

  // Region labeling specific hotkeys
  if (textboxModeActive.value && !e.repeat) {
    if (key === 'e') {
      e.preventDefault()
      isEKeyPressed.value = true
    }
    return
  }

  // Edge editing specific hotkeys
  if (!textlineModeActive.value || e.repeat) return

  if (key === 'd') {
    e.preventDefault()
    isDKeyPressed.value = true
    resetSelection()
  }
  if (key === 'a') {
    e.preventDefault()
    isAKeyPressed.value = true
    hoveredNodesForMST.clear()
    resetSelection()
  }
}

const handleGlobalKeyUp = (e) => {
  const key = e.key.toLowerCase()

  if (textboxModeActive.value && key === 'e') {
    isEKeyPressed.value = false
    textboxLabels.value++ // Increment label for the next group
  }

  if (!textlineModeActive.value) return

  if (key === 'd') isDKeyPressed.value = false
  if (key === 'a') {
    isAKeyPressed.value = false
    if (hoveredNodesForMST.size >= 2) addMSTEdges()
    hoveredNodesForMST.clear()
  }
}

const edgeExists = (nodeA, nodeB) =>
  workingGraph.edges.some(
    (e) =>
      (e.source === nodeA && e.target === nodeB) || (e.source === nodeB && e.target === nodeA)
  )
const addEdge = () => {
  if (selectedNodes.value.length !== 2 || edgeExists(...selectedNodes.value)) return
  const [source, target] = selectedNodes.value
  const newEdge = { source, target, label: 0, modified: true }
  workingGraph.edges.push(newEdge)
  modifications.value.push({ type: 'add', source, target, label: 0 })
  resetSelection()
}
const deleteEdge = () => {
  if (selectedNodes.value.length !== 2) return
  const [source, target] = selectedNodes.value
  const edgeIndex = workingGraph.edges.findIndex(
    (e) =>
      (e.source === source && e.target === target) || (e.source === target && e.target === source)
  )
  if (edgeIndex === -1) return
  const removedEdge = workingGraph.edges.splice(edgeIndex, 1)[0]
  modifications.value.push({
    type: 'delete',
    source: removedEdge.source,
    target: removedEdge.target,
    label: removedEdge.label,
  })
  resetSelection()
}
const undoModification = (index) => {
  const mod = modifications.value.splice(index, 1)[0]
  if (mod.type === 'add') {
    const edgeIndex = workingGraph.edges.findIndex(
      (e) => e.source === mod.source && e.target === mod.target
    )
    if (edgeIndex !== -1) workingGraph.edges.splice(edgeIndex, 1)
  } else if (mod.type === 'delete') {
    workingGraph.edges.push({
      source: mod.source,
      target: mod.target,
      label: mod.label,
      modified: true,
    })
  }
}
const resetModifications = () => {
  resetWorkingGraph()
  modifications.value = []
}

const distanceToLineSegment = (px, py, x1, y1, x2, y2) =>
  Math.hypot(
    px -
      (x1 +
        Math.max(
          0,
          Math.min(
            1,
            ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) /
              (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1)
          )
        ) *
          (x2 - x1)),
    py -
      (y1 +
        Math.max(
          0,
          Math.min(
            1,
            ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) /
              (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1)
          )
        ) *
          (y2 - y1))
  )
const handleEdgeHoverDelete = (mouseX, mouseY) => {
  for (let i = workingGraph.edges.length - 1; i >= 0; i--) {
    const edge = workingGraph.edges[i]
    const n1 = workingGraph.nodes[edge.source],
      n2 = workingGraph.nodes[edge.target]
    if (
      n1 &&
      n2 &&
      distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) <
        EDGE_HOVER_THRESHOLD
    ) {
      const removed = workingGraph.edges.splice(i, 1)[0]
      modifications.value.push({
        type: 'delete',
        source: removed.source,
        target: removed.target,
        label: removed.label,
      })
    }
  }
}
const handleNodeHoverCollect = (mouseX, mouseY) => {
  workingGraph.nodes.forEach((node, index) => {
    if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS)
      hoveredNodesForMST.add(index)
  })
}
const calculateMST = (indices, nodes) => {
  const points = indices.map((i) => ({ ...nodes[i], originalIndex: i }))
  const edges = []
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) {
      edges.push({
        source: points[i].originalIndex,
        target: points[j].originalIndex,
        weight: Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y),
      })
    }
  edges.sort((a, b) => a.weight - b.weight)
  const parent = {}
  indices.forEach((i) => (parent[i] = i))
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  const union = (i, j) => {
    const rootI = find(i),
      rootJ = find(j)
    if (rootI !== rootJ) {
      parent[rootJ] = rootI
      return true
    }
    return false
  }
  return edges.filter((e) => union(e.source, e.target))
}
const addMSTEdges = () => {
  calculateMST(Array.from(hoveredNodesForMST), workingGraph.nodes).forEach((edge) => {
    if (!edgeExists(edge.source, edge.target)) {
      const newEdge = { source: edge.source, target: edge.target, label: 0, modified: true }
      workingGraph.edges.push(newEdge)
      modifications.value.push({ type: 'add', ...newEdge })
    }
  })
}

const saveGeneratedGraph = async (name, page, g) => {
  try {
    await fetch(`${import.meta.env.VITE_BACKEND_URL}/save-graph/${name}/${page}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: g }),
    })
  } catch (e) {
    console.error('Error saving generated graph:', e)
  }
}

const saveModifications = async () => {
  const numNodes = workingGraph.nodes.length
  const labelsToSend = new Array(numNodes).fill(-1)
  for (const nodeIndex in textlineLabels) {
    labelsToSend[nodeIndex] = textlineLabels[nodeIndex]
  }

  const requestBody = {
    graph: workingGraph,
    modifications: modifications.value,
    textlineLabels: labelsToSend,
  }

  try {
    const res = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${localManuscriptName.value}/${localCurrentPage.value}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      }
    )
    if (!res.ok) throw new Error((await res.json()).error || 'Save failed')
    
    // Success handling
    const data = await res.json()
    graph.value = JSON.parse(JSON.stringify(workingGraph))
    modifications.value = []
    error.value = null
  } catch (err) {
    error.value = err.message
    throw err
  }
}

const saveCurrentGraph = async () => {
  if (isProcessingSave.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
    // alert('Graph and labels saved!')
  } catch (err) {
    alert(`Save failed: ${err.message}`)
  } finally {
    isProcessingSave.value = false
  }
}

const confirmAndNavigate = async (navAction) => {
  if (isProcessingSave.value) return
  if (modifications.value.length > 0) {
    if (confirm('You have unsaved changes. Do you want to save them before navigating?')) {
      isProcessingSave.value = true
      try {
        await saveModifications()
        navAction()
      } catch (err) {
        alert('Save failed, navigation cancelled.')
      } finally {
        isProcessingSave.value = false
      }
    } else {
      modifications.value = []
      navAction()
    }
  } else {
    navAction()
  }
}

const navigateToPage = (page) => {
  // If we are in the standalone/embedded flow, tell the parent to switch
  emit('page-changed', page)
}

const previousPage = () =>
  confirmAndNavigate(() => {
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex > 0) {
      navigateToPage(localPageList.value[currentIndex - 1])
    }
  })

const nextPage = () =>
  confirmAndNavigate(() => {
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex < localPageList.value.length - 1) {
      navigateToPage(localPageList.value[currentIndex + 1])
    }
  })
const goToIMG2TXTPage = () => {
  if (isEditModeFlow.value) {
    alert(
      "Text annotation is part of the 'New Manuscript' flow. This action is disabled in edit mode."
    )
    return
  }
  confirmAndNavigate(() => router.push({ name: 'img-2-txt' }))
}

const saveAndGoNext = async () => {
  if (loading.value || isProcessingSave.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex < localPageList.value.length - 1) {
      navigateToPage(localPageList.value[currentIndex + 1])
    } else {
      alert('This was the Last page. Saved successfully!')
    }
  } catch (err) {
    alert(`Save failed: ${err.message}`)
  } finally {
    isProcessingSave.value = false
  }
}

const runHeuristic = () => {
  if(!points.value.length) return;
  // Convert points format for LayoutGraphGenerator
  // Expected: [[x,y,s], ...]
  // We approximate size as 10 if not present, but usually 'points' has simple coords
  const rawPoints = points.value.map(p => [p.coordinates[0], p.coordinates[1], 10]); 
  
  const heuristicGraph = generateLayoutGraph(rawPoints);
  
  // Update workingGraph
  workingGraph.edges = heuristicGraph.edges.map(e => ({
     source: e.source, 
     target: e.target, 
     label: e.label, 
     modified: true 
  }));
  
  modifications.value.push({ type: 'reset_heuristic' }); // Marker for tracking
  computeTextlines();
}

onMounted(async () => {
  // Always respect props first in this semi-autonomous mode
  if (props.manuscriptName && props.pageName) {
    localManuscriptName.value = props.manuscriptName
    localCurrentPage.value = props.pageName
    await fetchPageList(props.manuscriptName)
    await fetchPageData(props.manuscriptName, props.pageName)
  }

  window.addEventListener('keydown', handleGlobalKeyDown)
  window.addEventListener('keyup', handleGlobalKeyUp)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleGlobalKeyDown)
  window.removeEventListener('keyup', handleGlobalKeyUp)
})

watch(
  () => props.pageName,
  (newPageName) => {
    if (newPageName && newPageName !== localCurrentPage.value) {
      localCurrentPage.value = newPageName
      fetchPageData(localManuscriptName.value, newPageName)
    }
  }
)

watch(textlineModeActive, (isEditing) => {
  if (isEditing) textboxModeActive.value = false
  if (!isEditing) {
    resetSelection()
    isAKeyPressed.value = false
    isDKeyPressed.value = false
    hoveredNodesForMST.clear()
  }
})

watch(textboxModeActive, (isLabeling) => {
  if (isLabeling) {
    console.log('Entering Region Labeling mode.')
    textlineModeActive.value = false
    resetSelection()

    // Ensure the next label index is unique by checking existing labels
    const existingLabels = Object.values(textlineLabels)
    if (existingLabels.length > 0) {
      // Find the maximum label value currently in use and add 1
      const maxLabel = Math.max(...existingLabels)
      textboxLabels.value = maxLabel + 1
      console.log(`Resuming labeling. Next available label index: ${textboxLabels.value}`)
    } else {
      // No labels exist yet, start from 0
      textboxLabels.value = 0
      console.log('No existing labels. Starting new labeling at index: 0')
    }
  } else {
    console.log('Exiting Region Labeling mode.')
  }
  hoveredTextlineId.value = null
})
</script>





<style scoped>
.manuscript-viewer {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  overflow: hidden;
  background-color: #333;
  color: #fff;
}

/* --- Toolbar --- */
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  background-color: #424242;
  border-bottom: 1px solid #555;
  flex-shrink: 0;
  gap: 16px;
}
.toolbar-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.toggle-container {
  display: flex;
  align-items: center;
  background-color: #3a3a3a;
  padding: 4px 8px;
  border-radius: 4px;
}

/* --- Main Visualization Area --- */
.visualization-container {
  position: relative;
  overflow: auto;
  flex-grow: 1;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 1rem;
}
.image-container {
  position: relative;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
}
.manuscript-image {
  display: block;
  user-select: none;
  opacity: 0.7;
}
.graph-overlay {
  position: absolute;
  top: 0;
  left: 0;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s ease-in-out;
}
.graph-overlay.is-visible {
  opacity: 1;
  pointer-events: auto;
}

/* --- Bottom Panel --- */
.bottom-panel {
  background-color: #4f4f4f;
  border-top: 1px solid #555;
  flex-shrink: 0;
  transition: all 0.3s ease;
}
.panel-toggle-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  cursor: pointer;
}
.edit-instructions p {
  margin: 0;
  font-size: 0.9em;
  color: #ccc;
  font-style: italic;
}
.bottom-panel-content {
  padding: 10px 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.edit-controls,
.modifications-log-container {
  display: flex;
  align-items: flex-start;
  gap: 20px;
}
.edit-actions {
  display: flex;
  gap: 8px;
}

/* --- UI Elements & States --- */
.panel-toggle-btn {
  padding: 4px 10px;
  font-size: 0.8em;
  background-color: #616161;
  border: 1px solid #757575;
}
.processing-save-notice,
.loading,
.error-message {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  padding: 20px 30px;
  border-radius: 8px;
  z-index: 10000;
  text-align: center;
}
.processing-save-notice {
  background-color: rgba(0, 0, 0, 0.8);
}
.error-message {
  background-color: #c62828;
}
.loading {
  font-size: 1.2rem;
  color: #aaa;
  background: none;
}
button {
  padding: 6px 14px;
  border-radius: 4px;
  border: 1px solid #666;
  background-color: #555;
  color: #fff;
  cursor: pointer;
  transition: background-color 0.2s ease;
}
button:hover:not(:disabled) {
  background-color: #6a6a6a;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* --- Modifications Log --- */
.modifications-details {
  flex-grow: 1;
}
.modifications-details h3 {
  margin: 0 0 8px 0;
  font-size: 1.1em;
  color: #eee;
}
.modifications-details ul {
  list-style-type: none;
  padding: 0;
  max-height: 120px;
  overflow-y: auto;
  border: 1px solid #666;
  background-color: #3e3e3e;
  border-radius: 3px;
}
.modification-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  border-bottom: 1px solid #555;
  font-size: 0.9em;
}
.modification-item:last-child {
  border-bottom: none;
}
.undo-button {
  background-color: #6d6d3d;
  border-color: #888855;
}
.undo-button:hover:not(:disabled) {
  background-color: #7a7a4a;
}
</style>
---
my-app/src/layout-analysis-utils/LayoutGraphGenerator.js
---
/**
 * Build a KD-Tree for fast neighbor lookup
 */
class KDTree {
  constructor(points) {
    this.points = points;
    this.tree = this.buildTree(points.map((p, i) => ({ point: p, index: i })), 0);
  }

  buildTree(points, depth) {
    if (points.length === 0) return null;
    if (points.length === 1) return points[0];

    const k = 2; // 2D points
    const axis = depth % k;
    
    points.sort((a, b) => a.point[axis] - b.point[axis]);
    const median = Math.floor(points.length / 2);
    
    return {
      point: points[median].point,
      index: points[median].index,
      left: this.buildTree(points.slice(0, median), depth + 1),
      right: this.buildTree(points.slice(median + 1), depth + 1),
      axis: axis
    };
  }

  query(queryPoint, k) {
    const best = [];
    
    const search = (node, depth) => {
      if (!node) return;
      
      const distance = this.euclideanDistance(queryPoint, node.point);
      
      if (best.length < k) {
        best.push({ distance, index: node.index });
        best.sort((a, b) => a.distance - b.distance);
      } else if (distance < best[best.length - 1].distance) {
        best[best.length - 1] = { distance, index: node.index };
        best.sort((a, b) => a.distance - b.distance);
      }
      
      const axis = depth % 2;
      const diff = queryPoint[axis] - node.point[axis];
      
      const closer = diff < 0 ? node.left : node.right;
      const farther = diff < 0 ? node.right : node.left;
      
      search(closer, depth + 1);
      
      if (best.length < k || Math.abs(diff) < best[best.length - 1].distance) {
        search(farther, depth + 1);
      }
    };
    
    search(this.tree, 0);
    return best.map(b => b.index);
  }

  euclideanDistance(p1, p2) {
    return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
  }
}

/**
 * DBSCAN clustering implementation to identify majority cluster and outliers
 */
function clusterWithSingleMajority(toCluster, eps = 10, minSamples = 2) {
  if (toCluster.length === 0) return [];
  
  // DBSCAN implementation
  const labels = dbscan(toCluster, eps, minSamples);
  
  // Count the occurrences of each label
  const labelCounts = {};
  labels.forEach(label => {
    labelCounts[label] = (labelCounts[label] || 0) + 1;
  });
  
  // Find the majority cluster label (excluding -1 outliers)
  let majorityLabel = null;
  let maxCount = 0;
  
  for (const [label, count] of Object.entries(labelCounts)) {
    const labelNum = parseInt(label);
    if (labelNum !== -1 && count > maxCount) {
      majorityLabel = labelNum;
      maxCount = count;
    }
  }
  
  // Create a new label array where the majority cluster is 0 and all others are -1
  const newLabels = new Array(labels.length).fill(-1); // Initialize all as outliers
  
  if (majorityLabel !== null) {
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === majorityLabel) {
        newLabels[i] = 0; // Assign 0 to the majority cluster
      }
    }
  }
  
  return newLabels;
}

/**
 * DBSCAN clustering algorithm implementation
 */
function dbscan(points, eps, minSamples) {
  const labels = new Array(points.length).fill(-1); // -1 means unclassified
  let clusterId = 0;
  
  for (let i = 0; i < points.length; i++) {
    if (labels[i] !== -1) continue; // Already processed
    
    const neighbors = getNeighbors(points, i, eps);
    
    if (neighbors.length < minSamples) {
      labels[i] = -1; // Mark as noise/outlier
    } else {
      // Start a new cluster
      expandCluster(points, labels, i, neighbors, clusterId, eps, minSamples);
      clusterId++;
    }
  }
  
  return labels;
}

/**
 * Get neighbors within eps distance
 */
function getNeighbors(points, pointIndex, eps) {
  const neighbors = [];
  const point = points[pointIndex];
  
  for (let i = 0; i < points.length; i++) {
    if (euclideanDistance(point, points[i]) <= eps) {
      neighbors.push(i);
    }
  }
  
  return neighbors;
}

/**
 * Expand cluster by adding density-reachable points
 */
function expandCluster(points, labels, pointIndex, neighbors, clusterId, eps, minSamples) {
  labels[pointIndex] = clusterId;
  
  let i = 0;
  while (i < neighbors.length) {
    const neighborIndex = neighbors[i];
    
    if (labels[neighborIndex] === -1) {
      labels[neighborIndex] = clusterId;
      
      const neighborNeighbors = getNeighbors(points, neighborIndex, eps);
      if (neighborNeighbors.length >= minSamples) {
        // Add new neighbors to the list (union operation)
        for (const newNeighbor of neighborNeighbors) {
          if (!neighbors.includes(newNeighbor)) {
            neighbors.push(newNeighbor);
          }
        }
      }
    }
    
    i++;
  }
}

function euclideanDistance(p1, p2) {
  return Math.sqrt(p1.reduce((sum, val, i) => sum + (val - p2[i]) ** 2, 0));
}

/**
 * Generate a graph representation of text layout based on points.
 * This function implements the core layout analysis logic.
 */
export function generateLayoutGraph(points) { // TODO ADD FEATURES
  const NUM_NEIGHBOURS = 6;
  const cos_similarity_less_than = -0.8;
  
  // Build a KD-tree for fast neighbor lookup
  const tree = new KDTree(points);
  const indices = points.map((point, i) => tree.query(point, NUM_NEIGHBOURS));
  
  // Store graph edges and their properties
  const edges = [];
  const edgeProperties = [];
  
  // Process nearest neighbors
  for (let currentPointIndex = 0; currentPointIndex < indices.length; currentPointIndex++) {
    const nbrIndices = indices[currentPointIndex];
    const currentPoint = points[currentPointIndex];
    
    const normalizedPoints = nbrIndices.map(idx => [
      points[idx][0] - currentPoint[0],
      points[idx][1] - currentPoint[1]
    ]);
    
    const scalingFactor = Math.max(...normalizedPoints.flat().map(Math.abs)) || 1;
    const scaledPoints = normalizedPoints.map(np => [np[0] / scalingFactor, np[1] / scalingFactor]);
    
    // Create a list of relative neighbors with their global indices
    const relativeNeighbours = nbrIndices.map((globalIdx, i) => ({
      globalIdx,
      scaledPoint: scaledPoints[i],
      normalizedPoint: normalizedPoints[i]
    }));
    
    const filteredNeighbours = [];
    
    for (let i = 0; i < relativeNeighbours.length; i++) {
      for (let j = i + 1; j < relativeNeighbours.length; j++) {
        const neighbor1 = relativeNeighbours[i];
        const neighbor2 = relativeNeighbours[j];
        
        const norm1 = Math.sqrt(neighbor1.scaledPoint[0] ** 2 + neighbor1.scaledPoint[1] ** 2);
        const norm2 = Math.sqrt(neighbor2.scaledPoint[0] ** 2 + neighbor2.scaledPoint[1] ** 2);
        
        let cosSimilarity = 0.0;
        if (norm1 * norm2 !== 0) {
          const dotProduct = neighbor1.scaledPoint[0] * neighbor2.scaledPoint[0] + 
                           neighbor1.scaledPoint[1] * neighbor2.scaledPoint[1];
          cosSimilarity = dotProduct / (norm1 * norm2);
        }
        
        // Calculate non-normalized distances
        const norm1Real = Math.sqrt(neighbor1.normalizedPoint[0] ** 2 + neighbor1.normalizedPoint[1] ** 2);
        const norm2Real = Math.sqrt(neighbor2.normalizedPoint[0] ** 2 + neighbor2.normalizedPoint[1] ** 2);
        const totalLength = norm1Real + norm2Real;
        
        // Select pairs with angles close to 180 degrees (opposite directions)
        if (cosSimilarity < cos_similarity_less_than) {
          filteredNeighbours.push({
            neighbor1,
            neighbor2,
            totalLength,
            cosSimilarity
          });
        }
      }
    }
    
    if (filteredNeighbours.length > 0) {
      // Find the shortest total length pair
      const shortestPair = filteredNeighbours.reduce((min, curr) => 
        curr.totalLength < min.totalLength ? curr : min
      );
      
      const { neighbor1: connection1, neighbor2: connection2, totalLength, cosSimilarity } = shortestPair;
      
      // Calculate angles with x-axis
      const thetaA = Math.atan2(connection1.normalizedPoint[1], connection1.normalizedPoint[0]) * 180 / Math.PI;
      const thetaB = Math.atan2(connection2.normalizedPoint[1], connection2.normalizedPoint[0]) * 180 / Math.PI;
      
      // Add edges to the graph
      edges.push([currentPointIndex, connection1.globalIdx]);
      edges.push([currentPointIndex, connection2.globalIdx]);
      
      // Calculate feature values for clustering
      const yDiff1 = Math.abs(connection1.normalizedPoint[1]);
      const yDiff2 = Math.abs(connection2.normalizedPoint[1]);
      const avgYDiff = (yDiff1 + yDiff2) / 2;
      
      const xDiff1 = Math.abs(connection1.normalizedPoint[0]);
      const xDiff2 = Math.abs(connection2.normalizedPoint[0]);
      const avgXDiff = (xDiff1 + xDiff2) / 2;
      
      // Calculate aspect ratio (height/width)
      const aspectRatio = avgYDiff / Math.max(avgXDiff, 0.001);
      
      // Calculate vertical alignment consistency
      const vertConsistency = Math.abs(yDiff1 - yDiff2);
      
      // Store edge properties for clustering
      edgeProperties.push([
        totalLength,
        Math.abs(thetaA + thetaB),
        // aspectRatio,
        // vertConsistency,
        // avgYDiff
      ]);
    }
  }
  
  // Cluster the edges based on their properties
  const edgeLabels = clusterWithSingleMajority(edgeProperties);
  
  // Create a mask for edges that are not outliers (label != -1)
  const nonOutlierMask = edgeLabels.map(label => label !== -1);
  
  // Prepare the final graph structure
  const graphData = {
    nodes: points.map((point, i) => ({
      id: i,
      x: parseFloat(point[0]),
      y: parseFloat(point[1]),
      s: parseFloat(point[2]),
    })),
    edges: []
  };
  
  // Add edges with their labels, filtering out outliers
  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    // Determine the corresponding edge label using division by 2 (each edge appears twice)
    const labelIndex = Math.floor(i / 2);
    const edgeLabel = edgeLabels[labelIndex];
    
    // Only add the edge if it is not an outlier
    if (nonOutlierMask[labelIndex]) {
      graphData.edges.push({
        source: parseInt(edge[0]),
        target: parseInt(edge[1]),
        label: parseInt(edgeLabel)
      });
    }
  }
  
  return graphData;
}
---
my-app/src/main.ts
---
import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(createPinia())
app.use(router)

app.mount('#app')

---
my-app/src/router/index.ts
---
import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [],
})

export default router

---
my-app/src/stores/counter.ts
---
import { ref, computed } from 'vue'
import { defineStore } from 'pinia'

export const useCounterStore = defineStore('counter', () => {
  const count = ref(0)
  const doubleCount = computed(() => count.value * 2)
  function increment() {
    count.value++
  }

  return { count, doubleCount, increment }
})

---
my-app/tsconfig.app.json
---
{
  "extends": "@vue/tsconfig/tsconfig.dom.json",
  "include": ["env.d.ts", "src/**/*", "src/**/*.vue"],
  "exclude": ["src/**/__tests__/*"],
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.app.tsbuildinfo",

    "paths": {
      "@/*": ["./src/*"]
    }
  }
}

---
my-app/tsconfig.json
---
{
  "files": [],
  "references": [
    {
      "path": "./tsconfig.node.json"
    },
    {
      "path": "./tsconfig.app.json"
    }
  ]
}

---
my-app/tsconfig.node.json
---
{
  "extends": "@tsconfig/node24/tsconfig.json",
  "include": [
    "vite.config.*",
    "vitest.config.*",
    "cypress.config.*",
    "nightwatch.conf.*",
    "playwright.config.*",
    "eslint.config.*"
  ],
  "compilerOptions": {
    "noEmit": true,
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.node.tsbuildinfo",

    "module": "ESNext",
    "moduleResolution": "Bundler",
    "types": ["node"]
  }
}

---
my-app/vite.config.ts
---
import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
})

---
pretrained_gnn/gnn_preprocessing_v2.yaml
---
# ===================================================================
#             Dataset Creation Configuration
# ===================================================================

min_nodes_per_page: 10 # Skip pages with fewer nodes than this

# --- MODIFIED: Data splitting configuration for a fixed split ---
splitting:
  random_seed: 49 # Seed for shuffling the val/test data before splitting
  val_ratio: 0.99 # 95% of val_test_data_dir goes to validation, 5% to test

# ===================================================================
#                Input Graph Construction
# ===================================================================
input_graph:
  # --- Heuristic Graph (Optional) ---
  use_heuristic_graph: True
  heuristic_params:
    k: 10 # HEURISTIC_GRAPH_K
    cosine_sim_threshold: -0.8 # OPPOSITE_NEIGHBOR_COS_SIM_THRESHOLD

  # --- Additional Connectivity ---
  # Strategy to add more edges to ensure GT is a subset.
  # Options: "knn", "none" (future: "expander")
  connectivity:
    # strategies: ["knn"] # Example: using both strategies "knn", 
    strategies: ["angular_knn"] #angular_knn

    # Parameters for the knn strategy
    knn_params:
      k: 12 # K for KNN # we can only predict if this is a "true" super set. Make this big!

    # Parameters for the second_shortest_heuristic strategy
    second_shortest_params:
          k: 10 # Number of neighbors to consider for finding pairs
          cosine_sim_threshold: -0.8 # Collinearity threshold
          min_angle_degrees: 45 # The minimum angle between the 1st and 2nd best pairs' edges

    angular_knn_params:
      k: 50 # K for angular KNN
      sector_angle_degrees: 20 # Minimum angle between edges to consider them connected


  # --- Graph Structure ---
  # Options: "bidirectional", "unidirectional"
  directionality: "bidirectional"

# ===================================================================
#                  Feature Engineering
# ===================================================================
features:
  # --- Node Features ---
  use_node_coordinates: True
  use_node_font_size: False
  use_heuristic_degree: True
  heuristic_degree_encoding: "one_hot" # Options: "linear_map", "one_hot"
  heuristic_degree_encoding_params:
    linear_map_factor: 0.1
    one_hot_max_degree: 10 # Cap for one-hot encoding dimension

  # --- Edge Features ---
  use_relative_distance: True # rel_x, rel_y
  use_euclidean_distance: True
  use_aspect_ratio_rel: True
  use_overlap: True
  overlap_encoding: "one_hot" # Options: "linear_map", "one_hot"
  overlap_encoding_params:
    linear_map_factor: 0.1
    one_hot_max_overlap: 10

  # --- Graph-level Features ---
  use_page_aspect_ratio: True

# ===================================================================
#              Ground Truth Graph Construction
# ===================================================================
ground_truth:
  # Algorithm to connect nodes within a textline
  # Options: "mst", "greedy_path"
  algorithm: "mst"

# ===================================================================
#               Alternate (Sklearn) Dataset Format
# ===================================================================
sklearn_format:
  enabled: False
  features:
    # Select which features to include in the CSV
    - "source_node_features" # includes x, y, font_size, heuristic_degree
    - "target_node_features"
    - "edge_features" # includes all enabled edge features
    - "page_features" # includes page_aspect_ratio
  # Advanced: Add aggregated features from N-hop neighbors
  # This is computationally expensive.
  use_n_hop_features: False
  n_hop_config:
    hops: 1 # Number of hops
    aggregations: ["mean", "std"] # Aggregations to apply
---
README.md
---
## **Semi-Autonomous Mode (Human-in-the-Loop)**

To run the application with a user interface for verification and correction:


#### 1 Install Conda Environment
    ```bash
    cd app
    conda env create -f environment.yaml
    conda activate gnn_layout
    ```

#### 2 Start Backend Server
    ```bash
    cd app
    conda activate gnn_layout
    python app.py
    ```
    The server runs on `http://localhost:5000`.

#### 3 Start Frontend
    ```bash
    cd app/my-app
    npm install
    npm run dev
    ```
    Access the UI at `http://localhost:5173`.

---
segmentation/craft.py
---
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models
# import matplotlib.pyplot as plt
from collections import namedtuple
from packaging import version
from collections import OrderedDict


# #GLOBAL VARIABLES
# lineheight_baseline_percentile = None
# binarize_threshold = None




def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else: #torchvision.__version__ < 0.13
            models.vgg.model_urls['vgg16_bn'] = models.vgg.model_urls['vgg16_bn'].replace('https://', 'http://')
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try: # multi gpu needs this
            self.rnn.flatten_parameters()
        except: # quantization doesn't work with this
            pass
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)



class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction

"""### CRAFT Model"""

#CRAFT

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def detect(img, detector, device):
    x = [np.transpose(normalizeMeanVariance(img), (2, 0, 1))]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)
    with torch.no_grad():
        y = detector(x)
        
    region_score = y[0,:,:,0].cpu().data.numpy()
    affinity_score = y[0,:,:,1].cpu().data.numpy()

    # clear GPU memory
    del x
    del y
    torch.cuda.empty_cache()

    return region_score,affinity_score
---
segmentation/pretrained_unet_craft/README.md
---
Download craft_mlt_pth from [here](https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth?download=true) into this folder.

Reference paper: https://arxiv.org/pdf/1904.01941
---
segmentation/segment_graph.py
---
import os
import numpy as np
import torch
import cv2
from scipy.ndimage import maximum_filter
from scipy.ndimage import label
from scipy.ndimage import maximum_filter, label
from skimage.draw import circle_perimeter

from .craft import CRAFT, copyStateDict, detect
from .utils import load_images_from_folder


def heatmap_to_pointcloud(heatmap, min_peak_value=0.3, min_distance=5, max_growth_radius=50):
    """
    Convert a 2D heatmap to a point cloud (X, Y, Radius) by identifying local maxima
    and estimating a radius for each by growing a circle as long as the heatmap
    intensity along the circumference is decreasing.
    
    Parameters:
    -----------
    heatmap : numpy.ndarray
        2D array representing the heatmap.
    min_peak_value : float
        Minimum normalized value for a peak to be considered (normalized between 0 and 1).
        Peaks must have an intensity strictly greater than this value.
    min_distance : int
        Minimum distance between peaks in pixels. Used for `maximum_filter`.
    max_growth_radius : int, optional
        Maximum radius the circle is allowed to grow. If None, it defaults to
        half the minimum dimension of the heatmap.
        
    Returns:
    --------
    points_with_radius : numpy.ndarray
        Array of shape (N, 3) where N is the number of detected characters.
        Each row contains [Peak_X, Peak_Y, Estimated_Radius].
        Peak_X, Peak_Y are from the original peak detection.
        Estimated_Radius is the radius of the largest circle around the peak
        for which the average intensity on its circumference was still decreasing.
    """
    if heatmap.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max == h_min: # Handle flat heatmap
        return np.empty((0, 3), dtype=np.float64)
        
    # 1. Normalize heatmap to [0, 1]
    heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    
    # 2. Find local maxima (Original logic)
    local_max_values = maximum_filter(heatmap_norm, size=min_distance)
    peaks_mask = (heatmap_norm == local_max_values) & (heatmap_norm > min_peak_value)
    
    # 3. Label connected components of these peak pixels
    labeled_individual_peaks, num_individual_peaks = label(peaks_mask)
    
    if num_individual_peaks == 0:
        return np.empty((0, 3), dtype=np.float64)

    points_and_radius = []
    
    H, W = heatmap_norm.shape
    if max_growth_radius is None:
        max_r_search = min(H, W) // 2
    else:
        max_r_search = max_growth_radius

    # 4. For each peak, grow a circle to estimate radius
    for peak_idx in range(1, num_individual_peaks + 1):
        peak_loc_y_arr, peak_loc_x_arr = np.where(labeled_individual_peaks == peak_idx)
        
        if peak_loc_y_arr.size == 0:
            continue
            
        peak_y, peak_x = peak_loc_y_arr[0], peak_loc_x_arr[0] # Use the first pixel of the peak area

        current_peak_intensity = heatmap_norm[peak_y, peak_x]
        last_ring_avg_intensity = current_peak_intensity
        estimated_radius = 0 # Radius 0 is the peak itself

        for r_test in range(1, max_r_search + 1):
            # Get coordinates of pixels on the circumference of radius r_test
            # skimage.draw.circle_perimeter ensures coordinates are within `shape` if provided.
            rr, cc = circle_perimeter(peak_y, peak_x, r_test, shape=heatmap_norm.shape)
            
            if rr.size == 0: # No pixels on this circumference (e.g., peak near edge, radius too large)
                break 

            current_ring_intensities = heatmap_norm[rr, cc]
            current_ring_avg_intensity = np.mean(current_ring_intensities)

            # Stop if slope is no longer strictly downward (i.e., current is flat or increasing)
            if current_ring_avg_intensity >= last_ring_avg_intensity:
                break 
            else:
                # Still decreasing, this radius is good. Update for next iteration.
                last_ring_avg_intensity = current_ring_avg_intensity
                estimated_radius = r_test # Update to this successful radius
        
        points_and_radius.append([float(peak_x), float(peak_y), float(estimated_radius)])

    return np.array(points_and_radius, dtype=np.float64)







def images2points(folder_path):
    print(folder_path)
    # how to get manuscript path from folder path - get parent directory
    m_path = os.path.dirname(folder_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Model Loading ---
    _detector = CRAFT()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(BASE_DIR, "pretrained_unet_craft", "craft_mlt_25k.pth")
    _detector.load_state_dict(copyStateDict(torch.load(pth_path, map_location=device)))
    detector = torch.nn.DataParallel(_detector).to(device)
    detector.eval()

    # --- Data Loading ---
    inp_images, file_names = load_images_from_folder(folder_path)
    print("Current Working Directory:", os.getcwd())

    # --- Processing Loop ---
    out_images = []
    normalized_points_list = [] # List for normalized points
    unnormalized_points_list = [] # NEW: List for raw, unnormalized points
    page_dimensions = []
    
    for image, _filename in zip(inp_images, file_names):
        # 0. Store original page dimensions
        original_height, original_width, _ = image.shape
        page_dimensions.append((original_width, original_height))

        # 1. Get region score (heatmap)
        region_score, affinity_score = detect(image, detector, device)
        assert region_score.shape == affinity_score.shape
        
        # 2. Convert heatmap to raw point coordinates (unnormalized)
        raw_points = heatmap_to_pointcloud(region_score, min_peak_value=0.4, min_distance=20)
        
        # --- NEW: Store the unnormalized points first ---
        unnormalized_points_list.append(raw_points)

        # 3. Normalize the points
        height, width = region_score.shape
        longest_dim = max(height, width)
        
        if longest_dim > 0:
            normalized_points = raw_points / longest_dim
        else:
            normalized_points = raw_points

        # 4. Store the processed data
        normalized_points_list.append(normalized_points)
        out_images.append(np.copy(region_score))

    # --- Saving Results ---
    heatmap_dir = f'{m_path}/heatmaps'
    base_data_dir = f'{m_path}/gnn-dataset'
    # frontend_graph_data_dir = f'{m_path}/frontend-graph-data'

    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(base_data_dir, exist_ok=True)
    # os.makedirs(frontend_graph_data_dir, exist_ok=True)

    # Save heatmaps
    for _img, _filename in zip(out_images, file_names):
        cv2.imwrite(os.path.join(heatmap_dir, _filename), 255 * _img)
    
    # --- Save NORMALIZED node features (for backward compatibility) ---
    for points, _filename in zip(normalized_points_list, file_names):
        output_filename = os.path.splitext(_filename)[0] + '_inputs_normalized.txt'
        output_path = os.path.join(base_data_dir, output_filename)
        np.savetxt(output_path, points, fmt='%f')

    # --- NEW: Save UNNORMALIZED node features to a separate file ---
    for raw_points, _filename in zip(unnormalized_points_list, file_names):
        raw_output_filename = os.path.splitext(_filename)[0] + '_inputs_unnormalized.txt'
        raw_output_path = os.path.join(base_data_dir, raw_output_filename)
        np.savetxt(raw_output_path, raw_points, fmt='%f')

    # Save the page dimensions
    for (width, height), _filename in zip(page_dimensions, file_names):
        dims_filename = os.path.splitext(_filename)[0] + '_dims.txt'
        dims_path = os.path.join(base_data_dir, dims_filename)
        with open(dims_path, 'w') as f:
            f.write(f"{width/2} {height/2}")


    # --- Cleanup ---
    del detector
    del _detector
    torch.cuda.empty_cache()

    print(f"Finished processing. All data saved to: {base_data_dir}")
    



---
segmentation/utils.py
---
import os
import numpy as np
import cv2
from skimage import io

# Function Definitions
def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def load_images_from_folder(folder_path):
    inp_images = []
    file_names = []
    
    # Get all files in the directory
    files = sorted(os.listdir(folder_path))
    
    for file in files:
        # Check if the file is an image (PNG or JPG)
        if file.lower().endswith(('.png', '.jpg', '.jpeg','.tif')):
            try:
                # Construct the full file path
                file_path = os.path.join(folder_path, file)
                
                # Open the image file
                image = loadImage(file_path)
                
                # Append the image and filename to our lists
                inp_images.append(image)
                file_names.append(file)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    return inp_images, file_names
---
segment_from_point_clusters.py
---
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




def load_node_features_and_labels(points_file, labels_file):
    points = np.loadtxt(points_file, dtype=float, ndmin=2).astype(int)
    with open(labels_file, "r") as f: labels = [line.strip() for line in f]
    features, filtered_labels = [], []
    for point, label in zip(points, labels):
        if label.lower() != "none":
            features.append(point)
            filtered_labels.append(int(label))
    return np.array(features), np.array(filtered_labels)

def assign_labels_and_plot(bounding_boxes, points, labels, image, output_path):
    # if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labeled_bboxes = []
    for x_min, y_min, w, h in bounding_boxes:
        x_max, y_max = x_min + w, y_min + h
        pts = [(p[0], p[1], lab) for p, lab in zip(points, labels) if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max]
        if pts and len({lab for _, _, lab in pts}) == 1:
            labeled_bboxes.append((x_min, y_min, w, h, pts[0][2]))
        elif pts:
            pts.sort(key=lambda p: p[1])
            boundaries = [y_min] + [max(y_min, min(y_max, int((pts[i-1][1] + pts[i][1]) / 2))) for i in range(1, len(pts)) if pts[i][2] != pts[i-1][2]] + [y_max]
            for i in range(1, len(boundaries)):
                top, bot = boundaries[i-1], boundaries[i]
                seg_label = next((lab for _, py, lab in pts if top <= py <= bot), None)
                if seg_label: labeled_bboxes.append((x_min, top, w, bot - top, seg_label))
    # for x, y, w, h, label in labeled_bboxes:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     cv2.putText(image, str(label), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imwrite(output_path, image)
    # print(f"Annotated image saved as: {output_path}")
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
---

