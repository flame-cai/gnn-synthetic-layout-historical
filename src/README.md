## Install Conda Environment

```bash
cd src
conda env create -f environment.yaml
conda activate gnn_layout
```

## Dataset Creation
#### Generate Synthetic Data
Configure the parameters in `synthetic_data_gen/configs/synthetic.yaml` as needed, then run:
```bash
cd src

python synthetic_data_gen/generate.py --dry-run --config synthetic_data_gen/configs/synthetic.yaml
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
--config synthetic_data_gen/configs/augment.yaml \
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
