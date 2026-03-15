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