# Installation Instructions

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