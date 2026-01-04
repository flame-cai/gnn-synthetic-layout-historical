## Install Conda Environment

```bash
cd src
conda env create -f environment.yaml
conda activate gnn_layout
```

## Dataset Creation
#### Generate Synthetic Data
Configure the parameters in `configs/synthetic.yaml` as needed, then run:
```bash
cd src

python synthetic_data_gen/generate.py --dry-run --config configs/synthetic. yaml  # to visualize a few samples
python synthetic_data_gen/generate.py --config configs/synthetic.yaml
```

This will create a new folder `src/gnn_data/synthetic_layout_data/` with all the generated synthetic data files in the graph based format.

This script peforms domain randomization to generate synthetic layout data simulating complex layouts in the graph based formulation introduced in this project. Both the synthetic data and the real data use the same graph based format, making it easy to integrate synthetic data into training pipelines.


#### To Flatten Sanskrit Dataset
The flattened sanskrit dataset is already provided in `src/gnn_data/flattened_sanskrit_data/` (without original images). However, if you wish to flatten the original hierarchical dataset again, with the original images and new feature engineering, you can run:

```bash
cd dataset

python flatten.py
```
This will create a new folder `src/gnn_data/flattened_sanskrit_data/` with all the data files flattened into a single directory structure, with an `index.csv`.


#### To Augment Sanskrit Dataset
```bash
cd src

python synthetic_data_gen/augment.py \
--config configs/augment.yaml \
--input_dir "gnn_data/flattened_sanskrit_data/gnn-dataset" \
--output_dir "gnn_data/augmented_sanskrit_dataset/"
```
This will create a new folder `src/gnn_data/augmented_sanskrit_dataset/` with three subfolders: `train`, `val` and `test`. `train` will contain the augmented training samples, while `val` and `test` will contain the original validation and test samples respectively.


#### Preprocess Data for GNN Training
First, copy synthetic data, augmented sanskrit data (training set) into a single folder. For example, you can create a new folder `src/gnn_data/combined_data/` and copy the following into it:
```bash
cd src
mkdir -p gnn_data/combined_data/
rsync -a gnn_data/generated_synthetic_data/ gnn_data/combined_data/
rsync -a gnn_data/augmented_sanskrit_dataset/train/ gnn_data/combined_data/
echo "augmented real data + synthetic data prepared at: gnn_data/combined_data/"
```

## Training and Evaluation
#### Prepare Data for GNN Training
First configure the data preprocessing parameters in `configs/gnn_preprocessing.yaml` as needed, then run:
```bash
cd src
python gnn_training/gnn_data_preparation/main_create_dataset.py \
--config configs/gnn_preprocessing.yaml \
--train_data_dir gnn_data/combined_data/ \
--val_test_data_dir gnn_data/augmented_sanskrit_dataset/val/ \
--output_dir gnn_data/processed_data_gnn/
```
This will create a new folder `src/gnn_data/processed_data_gnn/` with all the processed data files ready for GNN training (node features, edge features, labels etc.).

#### Train GNN Model
First configure the GNN training parameters in `configs/gnn_training.yaml` as needed, then run:
```bash

UNIQUE_FOLDER_NAME="gnn_experiment_1"  # change this to a unique name for each experiment
python -m gnn_training.training.main_train_eval \
--config "configs/gnn_training.yaml" \
--dataset_path "gnn_data/processed_data_gnn/" \
--unique_folder_name "${UNIQUE_FOLDER_NAME}" \
--gpu_id "${GPU_ID:-0}"
```
This will create a new folder `src/gnn_training/runs/${UNIQUE_FOLDER_NAME}/`.

