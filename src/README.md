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
python synthetic_data_gen/generate.py --dry-run --config synthetic_data_gen/configs/synthetic.yaml
```

This will create a new folder `src/gnn_data/synthetic_layout_data/` with all the generated synthetic data files in the graph based format.

This script peforms domain randomization to generate synthetic layout data simulating complex layouts in the graph based formulation introduced in this project. Both the synthetic data and the real data use the same graph based format, making it easy to integrate synthetic data into training pipelines.


#### To Flatten Sanskrit Dataset
The flattened sanskrit dataset is already provided in `src/gnn_data/flattened_sanskrit_dataset/` (without original images). However, if you wish to flatten the original hierarchical dataset again, with the original images and additional processing, you can run:

```bash
cd dataset
python flatten.py
```
This will create a new folder `src/gnn_data/flattened_sanskrit_dataset/` with all the data files flattened into a single directory structure, with an `index.csv`.

## Training and Evaluation
