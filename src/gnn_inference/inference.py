import os
import gc
from PIL import Image
import torch

# Assuming images2points is defined elsewhere or imported
# from your_module import images2points 
from segmentation.segment_graph import images2points
from inference_with_eval import run_inference_with_eval

import sys
# Get the directory where the current script is located (gnn_inference)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)
# Add 'src' to the system path so Python can find 'gnn_training'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print(f"Added {parent_dir} to sys.path to allow imports from 'gnn_training'")




def new_process_manuscript():
    MANUSCRIPTS_PATH = "./input_manuscripts"
    manuscript_name = "sample_manuscript_1"
    
    # Setup paths
    folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
    source_images_path = os.path.join(folder_path, "images")
    
    # We will save processed (and potentially resized) images here
    # to avoid modifying source files while iterating over them.
    resized_images_path = os.path.join(folder_path, "images_resized")

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


# in inference_with_eval.py
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run GNN pipeline for a full system-level evaluation.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("--input_dir", type=str, required=True, help="Directory with _inputs_normalized.txt and _labels_textline.txt files.")
#     parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results, visualizations, and logs.")
#     parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained .pt model checkpoint.")
#     parser.add_argument("--dataset_config_path", type=str, required=True, help="Path to the dataset creation config YAML. MUST match the one used for training.")
#     parser.add_argument("--visualize", action="store_true", help="Generate and save system-level evaluation visualizations.")
#     parser.add_argument("--BINARIZE_THRESHOLD", type=float, default=130, help="Threshold for binarizing the heatmap.")
#     parser.add_argument("--BBOX_PAD_V", type=float, default=0.7, help="Vertical padding for bounding boxes in line segmentation.")
#     parser.add_argument("--BBOX_PAD_H", type=float, default=0.5, help="Horizontal padding for bounding boxes in line segmentation.")
#     parser.add_argument("--CC_SIZE_THRESHOLD_RATIO", type=float, default=0.4, help="Connected component size threshold ratio for line segmentation.")
    
#     args = parser.parse_args()
#     run_inference_with_eval(args)


# args to pass:
#   --input_dir "/home/kartik/gnn_layout_project/data_processing/split_dataset/${UNIQUE_FOLDER_NAME}/proposed-method/test/gnn-dataset" \
#   --output_dir "${LATEST_MODEL_DIR}/${UNIQUE_FOLDER_NAME}" \
#   --model_checkpoint "/home/kartik/gnn_layout_project/models/proposed_method/runs/${UNIQUE_FOLDER_NAME}/${MODEL_ID}/best_model.pt" \
#   --dataset_config_path $CONFIG_PATH \
#   --BINARIZE_THRESHOLD 0.5098 \
#   --BBOX_PAD_V 0.7 \
#   --BBOX_PAD_H 0.5 \
#   --CC_SIZE_THRESHOLD_RATIO 0.4 \
#   --visualize
import argparse

# 1. Initialize the parser

if __name__ == "__main__":
    # 1. Parse standard CLI arguments4
    parser = argparse.ArgumentParser(description="GNN Layout Analysis Inference")
    args = parser.parse_args()
    
    # 2. Run the manuscript processing (images -> resized -> points)
    # This prepares the data needed for the inference step below
    new_process_manuscript()

    # 3. Override/Set specific parameters in the args object
    # -- Paths --
    args.input_dir = "./input_manuscripts/sample_manuscript_1/gnn-dataset"
    args.output_dir = "./input_manuscripts/sample_manuscript_1/evaluation_results"
    args.model_checkpoint = "/home/kartik/gnn_layout_analysis_publish/handwritten-sanskrit-layout-analysis-project/src/training_runs/gnn_experiment_1/SplineCNN-fold_0-20260104-132850/best_model.pt"
    args.dataset_config_path = "/home/kartik/gnn_layout_analysis_publish/handwritten-sanskrit-layout-analysis-project/src/configs/gnn_preprocessing.yaml"
    # -- Hyperparameters
    args.visualize = True
    args.BINARIZE_THRESHOLD = 0.5098
    args.BBOX_PAD_V = 0.7
    args.BBOX_PAD_H = 0.5
    args.CC_SIZE_THRESHOLD_RATIO = 0.4

    # 4. Run inference with the populated args object
    print(f"Starting inference with model: {os.path.basename(args.model_checkpoint)}")
    run_inference_with_eval(args)



