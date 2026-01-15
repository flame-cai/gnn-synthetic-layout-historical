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





def images2points(folder_path, min_distance=20):
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
        # --- MODIFIED: Use the passed min_distance parameter ---
        raw_points = heatmap_to_pointcloud(region_score, min_peak_value=0.4, min_distance=min_distance)
        
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

