import os
import shutil
import pandas as pd
import logging
import glob
from collections import defaultdict

# --- 1. Define Constants for Clarity and Maintenance ---
# Folders that are part of the flattened structure
FLATTENED_FOLDERS = ["page-xml-graph-groundtruth", "images", "heatmaps", "gnn-dataset"]

# Known suffixes for files within the gnn-dataset folder
# These are the 4 required files for the assertion check
GNN_REQUIRED_SUFFIXES = [
    "_dims",
    "_labels_textline",
    "_inputs_unnormalized",
    "_inputs_normalized"
]
# An optional file that should be copied but not counted in the main assertion
GNN_OPTIONAL_SUFFIX = "_labels_region"
ALL_GNN_SUFFIXES = GNN_REQUIRED_SUFFIXES + [GNN_OPTIONAL_SUFFIX,"_gt_stats", "_missed_lines"]


def get_base_page_name(filename: str, folder_name: str) -> str:
    """
    Robustly extracts the base page name from a filename.
    
    For files in 'gnn-dataset', it removes known suffixes. For all others,
    it simply removes the file extension. This avoids fragile split() logic.
    """
    base_name = filename.rsplit('.', 1)[0]
    
    if folder_name == 'gnn-dataset':
        for suffix in ALL_GNN_SUFFIXES:
            if base_name.endswith(suffix):
                # Remove the known suffix to get the true base name
                return base_name[:-len(suffix)]
    return base_name


def flatten_and_verify_directory(source_dir: str, dest_dir: str):
    """
    Flattens a nested directory, assigns unique zero-padded IDs, creates an index,
    and verifies the final file structure.

    Args:
        source_dir (str): The path to the root directory of the nested structure.
        dest_dir (str): The path to the destination directory for the flattened structure.
    """
    # --- 2. Setup Destination Directory and Logging ---
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as e:
        print(f"FATAL: Could not create destination directory {dest_dir}. Error: {e}")
        return

    log_file = os.path.join(dest_dir, "flattening.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    
    logging.info(f"Starting directory flattening. Source: '{source_dir}', Destination: '{dest_dir}'")

    # --- 3. Create Destination Sub-Directories ---
    try:
        for folder in FLATTENED_FOLDERS:
            os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)
        logging.info("Destination sub-directories created successfully.")
    except OSError as e:
        logging.error(f"Error creating destination sub-directories: {e}")
        return

    # --- 4. Initialize Core Data Structures ---
    index_data = []
    page_map = {}  # Maps original_unique_page_id -> short_id
    id_counter = 1

    # --- 5. Walk Through the Source Directory to Find and Process Files ---
    logging.info("--- Phase 1: Scanning source files and copying to destination ---")
    for dirpath, _, filenames in os.walk(source_dir):
        current_folder_name = os.path.basename(dirpath)
        if current_folder_name not in FLATTENED_FOLDERS:
            continue

        logging.info(f"Scanning folder: {dirpath}")
        
        # Determine dataset and sub_manuscript from the directory path relative to the source
        relative_path = os.path.relpath(dirpath, source_dir)
        path_parts = relative_path.split(os.sep)
        
        # The first part is always the dataset
        dataset = path_parts[0]
        # The last part is the folder type (images, heatmaps, etc.)
        # If there are more than 2 parts, the middle parts form the sub_manuscript_id
        sub_manuscript = os.sep.join(path_parts[1:-1]) if len(path_parts) > 2 else "NA"

        for filename in filenames:
            # Skip markdown files (documentation), they shouldn't be part of the flattened dataset
            if filename.lower().endswith('.md'):
                logging.info(f"Skipping markdown file: {os.path.join(dirpath, filename)}")
                continue

            # --- 6. Identify Original Page Name Robustly ---
            page_name_base = get_base_page_name(filename, current_folder_name)
            if not page_name_base:
                logging.warning(f"Could not determine base name for '{filename}' in '{dirpath}'. Skipping.")
                continue

            original_unique_page_id = f"{dataset}_{sub_manuscript}_{page_name_base}"

            # --- 7. Assign a New Short ID if this is the first time seeing this page ---
            if original_unique_page_id not in page_map:
                short_id = f"{id_counter:06d}"
                page_map[original_unique_page_id] = short_id
                
                index_data.append({
                    'short_id': short_id,
                    'original_unique_id': original_unique_page_id,
                    'dataset': dataset,
                    'sub_manuscript_id': sub_manuscript
                })
                id_counter += 1
            
            # --- 8. Construct New Filename and Copy the File ---
            assigned_short_id = page_map[original_unique_page_id]
            
            source_file = os.path.join(dirpath, filename)
            original_base_with_ext = filename.rsplit('.', 1)
            file_extension = original_base_with_ext[1]
            original_base_no_ext = original_base_with_ext[0]

            if current_folder_name == 'gnn-dataset':
                # For GNN, retain the suffix (e.g., _dims, _inputs_normalized)
                suffix_part = ""
                for suffix in ALL_GNN_SUFFIXES:
                    if original_base_no_ext.endswith(suffix):
                        suffix_part = suffix
                        break
                new_filename = f"{assigned_short_id}{suffix_part}.{file_extension}"
            else:
                new_filename = f"{assigned_short_id}.{file_extension}"
            
            dest_file = os.path.join(dest_dir, current_folder_name, new_filename)

            # --- 9. Assert No File Overwrites and Perform Copy ---
            assert not os.path.exists(dest_file), f"FATAL: Destination file '{dest_file}' already exists. This indicates a logic error in ID mapping."
            
            try:
                shutil.copy(source_file, dest_file)
            except Exception as e:
                logging.error(f"Error copying '{source_file}' to '{dest_file}': {e}")

    # --- 10. Post-Flattening Verification ---
    logging.info("--- Phase 2: Verifying the integrity of the flattened structure ---")
    all_checks_passed = True
    for entry in index_data:
        short_id = entry['short_id']
        
        try:
            # Check for 1 image file
            img_files = glob.glob(os.path.join(dest_dir, "images", f"{short_id}.*"))
            assert len(img_files) == 1, f"Expected 1 image file for {short_id}, found {len(img_files)}"

            # Check for 1 heatmap file
            heatmap_files = glob.glob(os.path.join(dest_dir, "heatmaps", f"{short_id}.*"))
            assert len(heatmap_files) == 1, f"Expected 1 heatmap file for {short_id}, found {len(heatmap_files)}"

            # Check for 1 XML file
            xml_files = glob.glob(os.path.join(dest_dir, "page-xml-graph-groundtruth", f"{short_id}.*"))
            assert len(xml_files) == 1, f"Expected 1 XML file for {short_id}, found {len(xml_files)}"
            
            # Check for exactly 4 required GNN files
            gnn_files = glob.glob(os.path.join(dest_dir, "gnn-dataset", f"{short_id}*"))
            # Exclude the optional file from the count for the assertion
            required_gnn_files = [f for f in gnn_files if not f.
                                  replace('.txt','').endswith(GNN_OPTIONAL_SUFFIX)]
            assert len(required_gnn_files) == 4, f"Expected 4 required GNN files for {short_id}, found {len(required_gnn_files)}"

            logging.info(f"Verification for {short_id}: PASSED (1x1x1x4 files confirmed)")

        except AssertionError as e:
            logging.error(f"Verification for {short_id}: FAILED - {e}")
            all_checks_passed = False

    if all_checks_passed:
        logging.info("--- All unique IDs passed the 1x1x1x4 verification successfully! ---")
    else:
        logging.warning("--- One or more IDs failed verification. Please review the log for errors. ---")

    # --- 11. Create the Final index.csv File ---
    if not index_data:
        logging.warning("No files were processed. The index.csv will be empty.")
        return

    try:
        index_df = pd.DataFrame(index_data)
        # cols = ['short_id', 'original_unique_id', 'dataset', 'sub_manuscript_id']
        cols = ['short_id','dataset']
        index_df = index_df[cols]
        index_df.to_csv(os.path.join(dest_dir, "index.csv"), index=False)
        logging.info(f"index.csv file created successfully with {len(index_df)} unique page entries.")
    except Exception as e:
        logging.error(f"Failed to create the final index.csv: {e}")

    logging.info("Directory flattening process completed.")


if __name__ == '__main__':
    
    source_directory = "."
    destination_directory = "../src/gnn_data/flattened_sanskrit_data"
    
    # Ensure source exists before running
    if not os.path.isdir(source_directory):
        print(f"FATAL: Source directory not found at '{source_directory}'")
    else:
        flatten_and_verify_directory(source_directory, destination_directory)