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