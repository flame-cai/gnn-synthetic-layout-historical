import os
import argparse
import gc
from PIL import Image
import torch

from segmentation.segment_graph import images2points
from gnn_inference import run_gnn_inference

import sys
# Get the directory where the current script is located (gnn_inference)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)
# Add 'src' to the system path so Python can find 'gnn_training'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print(f"Added {parent_dir} to sys.path to allow imports from 'gnn_training'")




def process_new_manuscript(manuscript_path="./input_manuscripts/sample_manuscript_1"):
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
                
                # --- MODIFICATION START ---
                # Check if the image is too big (height or width > 3000 pixels)
                width, height = image.size
                
                if width > 3000 or height > 3000:
                    print(f"Image '{filename}' is too large ({width}x{height}). Downscaling by 50%.")
                    new_width = width // 2
                    new_height = height // 2
                    
                    # Handle Resampling filter compatibility for older/newer PIL versions
                    try:
                        resampling_filter = Image.Resampling.LANCZOS
                    except AttributeError:
                        resampling_filter = Image.LANCZOS

                    image = image.resize((new_width, new_height), resampling_filter)
                # --- MODIFICATION END ---

                # Standardize Color Mode
                if image.mode in ("RGBA", "P", "LA"):
                    image = image.convert("RGB")

                # Save processed image to the NEW folder
                new_filename = f"{base_filename}.jpg"
                save_path = os.path.join(resized_images_path, new_filename)
                
                # We copy/save distinct files even if not resized to ensure 
                # images2points has a single complete directory to work with.
                image.save(save_path, "JPEG")
                print(f"Processed: {new_filename}")

        except Exception as img_err:
            print(f"Failed to process image {filename}: {img_err}")
            continue

    # Point the inference function to the new resized/processed folder
    print("Running images2points on processed folder...")
    images2points(resized_images_path) 
    
    # Cleanup resources
    torch.cuda.empty_cache()
    gc.collect()

    print("Processing complete.")






if __name__ == "__main__":
    # 1. Parse standard CLI arguments4
    parser = argparse.ArgumentParser(description="GNN Layout Analysis Inference")
    parser.add_argument("--manuscript_path", type=str, default="./input_manuscripts/sample_manuscript_1", help="Path to the manuscript directory")
    args = parser.parse_args()

    # the data preparation.yaml is tied to the model_checkpoint used.
    args.model_checkpoint = "./pretrained_gnn/best_model.pt"
    args.dataset_config_path = "./pretrained_gnn/gnn_preprocessing.yaml"

    # -- Hyperparameters
    args.visualize = True
    args.BINARIZE_THRESHOLD = 0.5098
    args.BBOX_PAD_V = 0.7
    args.BBOX_PAD_H = 0.5
    args.CC_SIZE_THRESHOLD_RATIO = 0.4

    process_new_manuscript(args.manuscript_path)
    run_gnn_inference(args)



