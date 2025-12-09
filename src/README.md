# Synthetic Data Generator
This script peforms domain randomization to generate synthetic layout data simulating complex layouts in the graph based formulation introduced in this project. Both the synthetic data and the real data use the same graph based format, making it easy to integrate synthetic data into training pipelines.

### Install Conda Environment

```bash
cd src
conda env create -f environment.yaml
conda activate gnn_layout
```

### Generate Synthetic Data
Configure the parameters in `synthetic_data_gen/configs/synthetic.yaml` as needed, then run:
```bash
python synthetic_data_gen/generate.py --dry-run --config synthetic_data_gen/configs/synthetic.yaml
```

This will create a new folder `src/gnn_data/synthetic_layout_data/` with all the generated synthetic data files in the graph based format.