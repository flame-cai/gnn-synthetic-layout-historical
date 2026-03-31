# AGENTS.md

This file is for coding agents working in this repository. It explains the paper-aligned mental model, maps the paper to the codebase, and gives a practical guide for setting up and using the project in automatic and semi-automatic modes.

## 1. What This Repository Implements

Primary paper:
- `chincholikar26towards.pdf`
- Title: `Towards Text-Line Segmentation of Historical Documents Using Graph Neural Networks`
- Authors: `Kartik Chincholikar`, `Kaushik Gopalan`, and `Mihir Hasabnis`
- Venue: `ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling`

High-level problem:
- Segment text lines from historical manuscript pages.
- Support difficult layouts: curved lines, marginalia, interlinear glosses, dense pages, and manuscript-to-manuscript distribution shifts.

Core idea from the paper:
- Represent each detected character or grapheme cluster as a graph node.
- Represent candidate relationships between characters as graph edges.
- Predict which candidate edges should be kept, turning text-line segmentation into binary edge classification.

Pipeline mental model:
1. Use CRAFT to detect character regions and produce a heatmap.
2. Convert the heatmap to a point cloud of character centers and radii.
3. Build a heuristic graph using geometric priors:
   - characters on a line should usually have two opposite neighbors
   - character spacing is usually smaller than line spacing
4. Add extra candidate connectivity so true text-line edges are available to the model.
5. Build node and edge features from geometry plus heuristic metadata.
6. Train a GNN to classify candidate edges as keep/delete.
7. Convert kept edges into connected components, which are the predicted text lines.
8. Export predictions as graph labels, PAGE-XML, and cropped line images for OCR.

Branching in the paper:
- Branch 1: heuristic graph + DBSCAN-style anomaly filtering + human correction for semi-automatic annotation.
- Branch 2: learned GNN edge classification for automatic prediction.

Important current-repo drift from the paper:
- The formulation is still the same, but some default configs have evolved.
- The paper describes KNN-12 additional connectivity and 50K synthetic samples.
- The current defaults use `angular_knn` in preprocessing and `40000` synthetic samples.
- The paper reports 30 epochs / 6 warmup epochs / early stopping 15.
- Current defaults are 10 epochs / 5 warmup epochs / early stopping 10.
- Treat `src/` as the main research pipeline and `app/` as the productized UI/inference layer.

## 2. Paper -> Code Map

### Research pipeline in `src/`

Synthetic data generation:
- `src/synthetic_data_gen/generate.py`
- `src/synthetic_data_gen/manuscript_generator/generator.py`
- `src/synthetic_data_gen/manuscript_generator/`

Real-data augmentation:
- `src/synthetic_data_gen/augment.py`

Dataset preprocessing for GNN training:
- `src/gnn_training/gnn_data_preparation/main_create_dataset.py`
- `src/gnn_training/gnn_data_preparation/graph_constructor.py`
- `src/gnn_training/gnn_data_preparation/feature_engineering.py`
- `src/gnn_training/gnn_data_preparation/dataset_generator.py`
- `src/configs/gnn_preprocessing.yaml`

Training loop:
- `src/gnn_training/training/main_train_eval.py`
- `src/gnn_training/training/engine.py`
- `src/gnn_training/training/metrics.py`
- `src/configs/gnn_training.yaml`

Model implementations:
- `src/gnn_training/training/models/gnn_models.py`
- Architectures present: `GCN`, `GAT`, `MPNN`, `SGC`, `SplineCNN`

Inference pipeline:
- `src/gnn_inference/inference.py`
- `src/gnn_inference/segmentation/segment_graph.py`
- `src/gnn_inference/gnn_inference.py`
- `src/gnn_inference/pretrained_gnn/`

### Application layer in `app/`

Backend:
- `app/app.py`
- `app/inference.py`
- `app/gnn_inference.py`
- `app/segmentation/segment_graph.py`

Frontend:
- `app/my-app/`

Semi-automatic annotation logic:
- UI-side graph generation and DBSCAN-style majority cluster filtering are in:
  - `app/my-app/src/layout-analysis-utils/LayoutGraphGenerator.js`

### What each core module does

Training loop:
- `src/gnn_training/training/main_train_eval.py`
- Loads processed datasets, creates the model, applies focal loss or weighted loss, runs warmup, early stopping, validation, and test evaluation.

Model:
- `src/gnn_training/training/models/gnn_models.py`
- Wraps GNN backbones with an edge-classification head.

Data pipeline:
- `src/gnn_training/gnn_data_preparation/main_create_dataset.py`
- `src/gnn_training/gnn_data_preparation/graph_constructor.py`
- `src/gnn_training/gnn_data_preparation/feature_engineering.py`
- Turns raw graph-format files into PyTorch Geometric datasets.

Evaluation:
- `src/gnn_training/training/engine.py`
- `src/gnn_training/training/metrics.py`
- Computes edge-level metrics and textline-level connected-component matching metrics.

Inference:
- `src/gnn_inference/inference.py`
- `src/gnn_inference/gnn_inference.py`
- Runs image preprocessing, CRAFT point extraction, graph construction, GNN inference, connected components, PAGE-XML generation, and line-image export.

## 3. Recommended Agent Mental Model

When working in this repo, think in terms of four layers:

1. Detection layer
- CRAFT turns page images into character heatmaps and point clouds.

2. Graph construction layer
- Heuristic graph edges encode geometry priors.
- Extra connectivity ensures the GNN can recover the true structure.

3. Learning layer
- The GNN predicts which candidate edges are true text-line links.

4. Output layer
- Connected components become text lines.
- Outputs are graph labels, XML, and cropped line images.

If the user asks whether something is "paper aligned," check:
- graph construction strategy
- synthetic-data scale
- training hyperparameters
- evaluation metric choice
- whether the code path is in `src/` or `app/`

## 4. Clone and Setup Guide

If the repo is not present locally, clone it:

```bash
git clone --depth 1 https://github.com/flame-cai/gnn-synthetic-layout-historical.git
cd gnn-synthetic-layout-historical
```

### Conda environment

Install Conda first if it is not already present, then create and install the Python environment:

```bash
conda create -n gnn_layout python=3.11 -y
conda activate gnn_layout
pip install -r requirements.txt
```

OS-specific activation:
- Windows PowerShell: `conda activate gnn_layout`
- Windows Command Prompt: `conda activate gnn_layout`
- macOS/Linux bash/zsh: `conda activate gnn_layout`

If `conda activate` fails:
- run `conda init powershell` on Windows PowerShell, then restart the shell
- run `conda init bash` or `conda init zsh` on macOS/Linux, then restart the shell

### Node / npm for the frontend

Install Node.js and npm from the official Node.js download page if not already present.

Verify:

```bash
node -v
npm -v
```

Then install frontend dependencies:

```bash
cd app/my-app
npm install
```

`npm install` only needs to be run once for the first setup. Later frontend launches usually only need `npm run dev`.

Frontend env file:

Create `app/my-app/.env` with:

```env
VITE_BACKEND_URL="http://localhost:5000"
```

### Optional OCR model setup

The app supports Gemini or a local OCR model.

### For the local pretrained Sanskrit specific OCR model, the repo expects:
- `app/recognition/pretrained_model/vadakautuhala.pth`

If it is missing, local OCR will not work until the user provides it. You will need to download the model as follows (or use your own finetuned one for other scripts)
```bash
cd app/recognition/pretrained_model
wget "https://docs.google.com/uc?export=download&id=1Mm0Keee3DQ4JY8Fe62zgBfRohdEHrfTk" -O vadakautuhala.pth
```

This `vadakautuhala.pth` model is specialized for a Sanskrit manuscript writing style from the Lalchand Research Library collection. README guidance says manuscript-specific fine-tuning is often beneficial.

Do not silently invent or substitute model weights.


### If the user wants Gemini-based text recognition:
- never fabricate an API key
- explicitly ask the user to provide their own key
- explain where it will be stored
- note that README guidance suggests adjusting the prompt in `_run_gemini_recognition_internal` in `app/app.py` for the manuscript language/script
- note that the current default Gemini model string in `app/app.py` is `genai.GenerativeModel('gemini-2.5-flash')`, which can be updated

Expected location:
- `app/.env`

Expected format:

```env
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

Agent behavior:
- prompt the user before creating or editing `app/.env`
- do not commit the key
- do not echo the key back in logs or responses
- if the key is missing, continue with layout analysis and explain that Gemini OCR is unavailable until the user provides it

## 5. Automatic Mode

Automatic mode means fully automatic layout analysis without manual graph editing.

Recommended research entry point:

```bash
cd src/gnn_inference
conda activate gnn_layout
python inference.py --manuscript_path "./demo_manuscripts/sample_manuscript_1/"
```

What this does:
- resizes large input images
- runs CRAFT
- creates point clouds
- runs the pretrained GNN
- writes outputs under `layout_analysis_output/`

Expected outputs:
- graph-format predictions
- PAGE-XML files
- segmented line images

README-specific operating notes:
- `sample_manuscript_1/` and `sample_manuscript_2/` are expected to work out of the box.
- `sample_manuscript_3/` is lower resolution and may require changing `min_distance` from `20` to `10` in `src/gnn_inference/segmentation/segment_graph.py`.
- The automatic inference path resizes very large images to `2500` pixels on the longest side in `src/gnn_inference/inference.py`.
- If that resize limit is changed, the `min_distance` feature-extraction setting should usually be re-tuned as well.
- The README explicitly frames out-of-the-box inference as most suitable when CRAFT can detect the script characters and character spacing is smaller than line spacing.

For app-managed automatic preprocessing on user-uploaded manuscripts:
- the backend route in `app/app.py` handles upload and preprocessing
- the actual preprocessing helper is `app/inference.py`

## 6. Semi-Automatic Mode

Semi-automatic mode means:
- run the backend
- run the frontend
- upload manuscript images
- let the tool generate an initial graph/prediction
- manually correct nodes/edges or text-box structure in the UI if needed

Start backend:

```bash
cd app
conda activate gnn_layout
python app.py
```

Start frontend in another shell:

```bash
cd app/my-app
npm run dev
```

Default URLs:
- backend: `http://localhost:5000`
- frontend: `http://localhost:5173`

The semi-automatic experience relies on:
- backend API in `app/app.py`
- frontend graph editing in `app/my-app`
- heuristic + DBSCAN-style graph proposal logic in `app/my-app/src/layout-analysis-utils/LayoutGraphGenerator.js`

## 8. Cross-Platform Notes

Windows:
- Prefer PowerShell commands and Windows path separators when giving examples to Windows users.
- `wget` may not be available; use browser download, `curl -L -o`, or PowerShell alternatives.
- `rsync` commands in the README are Unix-oriented. On Windows, use File Explorer, `Copy-Item`, or a Python copy script if absolutely necessary.

macOS/Linux:
- Shell examples from the README will usually work directly.
- Ensure `conda` is initialized in the shell.

GPU considerations:
- If CUDA is available, the code will usually use it automatically.
- If CUDA is unavailable, tell the user inference and training may be much slower.
- Do not assume multi-GPU support is stable everywhere just because a helper exists.

## 9. Edge Cases Agents Should Handle

Agents should proactively check for these issues and guide the user clearly:

- Missing `conda`, `node`, or `npm`
- Missing pretrained weights:
  - `app/pretrained_gnn/v2.pt`
  - `src/gnn_inference/pretrained_gnn/v2.pt`
  - CRAFT weights
  - optional OCR weights
- Missing frontend env file `app/my-app/.env`
- Missing Gemini key in `app/.env`
- Port conflicts on `5000` or `5173`
- Very low-resolution images
- Extremely large images that need resizing
- Different optimal `min_distance` for different manuscript resolutions
- CPU-only environments
- Windows users trying to run Unix-only commands from the README

Resolution-specific note:
- Lower-resolution manuscripts may need a smaller `min_distance` in point extraction.
- The README specifically calls out changing it from `20` to `10` for some low-resolution cases.

User experience guidance for agents:
- Prefer automatic mode first if the user wants quick results.
- Suggest semi-automatic mode if automatic results are poor on complex layouts.
- When OCR is optional, separate layout-analysis success from OCR setup success.
- If a sub-step fails, explain the exact blocker and the next best fallback.

## 10. Practical Command Reference

Install Python dependencies:

```bash
conda create -n gnn_layout python=3.11 -y
conda activate gnn_layout
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd app/my-app
npm install
```

Run backend:

```bash
cd app
conda activate gnn_layout
python app.py
```

Run frontend:

```bash
cd app/my-app
npm run dev
```

Run automatic sample inference:

```bash
cd src/gnn_inference
conda activate gnn_layout
python inference.py --manuscript_path "./demo_manuscripts/sample_manuscript_1/"
```

Generate synthetic data:

```bash
cd src
python synthetic_data_gen/generate.py --dry-run --config configs/synthetic.yaml
python synthetic_data_gen/generate.py --config configs/synthetic.yaml
```

Default synthetic-data output note:
- `src/configs/synthetic.yaml` currently writes to `src/gnn_data/generated_synthetic_data/`.
- README prose also mentions `src/gnn_data/synthetic_layout_data/`, but the current config default is `generated_synthetic_data`; agents should verify the configured `output_dir` before giving path-specific instructions.

Augment real data:

```bash
cd src
python synthetic_data_gen/augment.py --config configs/augment.yaml --input_dir "gnn_data/flattened_sanskrit_data/gnn-dataset" --output_dir "gnn_data/augmented_sanskrit_dataset/"
```

Create combined dataset:

```bash
cd src
mkdir -p gnn_data/combined_data/
rsync -a gnn_data/generated_synthetic_data/ gnn_data/combined_data/
rsync -a gnn_data/augmented_sanskrit_dataset/train/ gnn_data/combined_data/
```

Windows note:
- The README uses `rsync`, which is Unix-oriented. On Windows, prefer `Copy-Item`, File Explorer, or another non-`rsync` copy method.

Create processed GNN dataset:

```bash
cd src
python gnn_training/gnn_data_preparation/main_create_dataset.py --config configs/gnn_preprocessing.yaml --train_data_dir gnn_data/combined_data/ --val_test_data_dir gnn_data/augmented_sanskrit_dataset/val/ --output_dir gnn_data/processed_data_gnn/
```

Train the model:

```bash
cd src
python -m gnn_training.training.main_train_eval --config "configs/gnn_training.yaml" --dataset_path "gnn_data/processed_data_gnn/" --unique_folder_name "gnn_experiment_1" --gpu_id 0
```

## 11. Final Guidance for Agents

Default working assumptions:
- `src/` is the authoritative research implementation.
- `app/` is the UI and deployment layer.
- The repo is broadly aligned with the paper but not frozen to the exact experimental defaults reported there.

When asked to align code with the paper:
- compare current config defaults against the paper before changing code
- distinguish architectural mismatches from harmless config drift
- do not assume every duplicated module under `app/` and `src/` should be edited together unless the user wants both kept in sync

When asked to operate the application:
- set up Python first
- set up frontend second
- verify weights and env files
- only ask for Gemini API credentials if the user explicitly wants Gemini OCR
