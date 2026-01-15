# Towards Text-Line Segmentation of Historical Documents Using Graph Neural Networks and Synthetic Layout Data


**Version:** 3.0
**Last Updated:** Jan 14, 2026

## **Project Components**
*   **🧩 [Semi-Automatic Annotation Tool](https://github.com/flame-cai/gnn-synthetic-layout-historical?tab=readme-ov-file#-semi-automatic-annotation-tool):** Segment text-lines from complex layouts using Graph Neural Networks, followed by manual corrections to the output if required - supporting annotations at charcater level, text-line level and text-box level.
*   **💻 [Automatic Out-of-the-box Inference](https://github.com/flame-cai/gnn-synthetic-layout-historical?tab=readme-ov-file#-automatic-out-of-the-box-inference):** Run fully automatic stand-alone inference
*   **🧠 [GNN Training Recipe](https://github.com/flame-cai/gnn-synthetic-layout-historical?tab=readme-ov-file#gnn-training-recipe):** Train custom GNN architectures using synthetic data, augmented real data.
_________
*   **📁 [Dataset](https://github.com/flame-cai/gnn-synthetic-layout-historical/tree/main/dataset):** 15 Sanskrit Manuscripts, 481 pages, with diverse layouts, annotated in graph based and PAGE-XML format
*   **⚙️ [Synthetic Data Generator](https://github.com/flame-cai/gnn-synthetic-layout-historical?tab=readme-ov-file#-generate-synthetic-data):** Generate synthetic layout data simulating complex layouts in the graph based format


## **Semi-Automatic Annotation Tool**
This mode allows users to manually correct and refine the GNN-predicted layouts using an intuitive web-based interface. Users can adjust text-line connections, label text boxes, and modify node placements to ensure high-quality layout annotations.
![GNN Layout UI Demo](./app/demo_tutorial.gif)

### Setup Instructions
#### 1 Install Conda Environment
Install [Conda](https://docs.conda.io/en/latest/miniconda.html) first, then run:

    ```bash
    cd app
    conda env create -f environment.yaml
    conda activate gnn_layout
    ```

#### 2 Start Backend Server
    ```bash
    cd app
    conda activate gnn_layout
    python app.py
    ```
    The server runs on `http://localhost:5000`.

#### 3 Start Frontend
First install npm from [Node.js official website](https://nodejs.org/en/download/). 

Create a .env file in `src/app/my-app/` with the following content:

    ```env
    VITE_BACKEND_URL="http://localhost:5000"
    ```

Then run:

    ```bash
    cd app/my-app
    npm install
    npm run dev
    ```
    Access the UI at `http://localhost:5173`.



##  **Automatic Out-of-the-box Inference**
Run the entire layout analysis pipeline in fully automatic mode on sample manuscripts, to obtain text-line segmented images in PAGE-XML format, GNN format, and as individual line images.

#### 🔵 Install Conda Environment
```bash
cd src
conda env create -f environment.yaml
conda activate gnn_layout
```
#### 🔵 Run Inference (fully automatic)
```bash
cd src/gnn_inference
python inference.py --manuscript_path "./demo_manuscripts/sample_manuscript_1/"
```

This will process all the manuscript images in sample_manuscript_1 and save the segmented line images in folder `sample_manuscript_1/layout_analysis_output/` in PAGE_XML format, GNN format, and as individual line images.

> **NOTE 1:**  
> This project is made for Handwritten Sanskrit Manuscripts in Devanagari script, however it will work reasonibly well on other scripts if they fit the following criteria:
> 1) [CRAFT](https://github.com/clovaai/CRAFT-pytorch) successfully detects the script characters  
> 2) Character spacing is less than Line spacing. 
>
> If the output is not satisfactory, please use the Semi-Autonomous Mode to make corrections (add/delete edges or nodes, label text boxes etc.)


> **NOTE 2:**  
> `sample_manuscript_1/` and `sample_manuscript_2` contain high resolution images and will work out of the box. However, `sample_manuscript_3/` contains lower resolution images - for whom the feature engineering parameter `min_distance` in `src/gnn_inference/segmentation/segment_graph.py` will need to be reduced from `20` to `10` as follows:
> ```python
> `raw_points = heatmap_to_pointcloud(region_score, min_peak_value=0.4, min_distance=10)`
> ```
> The inference code resizes very large images to `2500` longest side for processing to reduce the GPU memory requirements and to standardize the feature extraction process. If you wish to change this limit, you can do so in `src/gnn_inference/inference.py` at the following lines:
> ```python
> target_longest_side = 2500
> ```
> However, this is also require adjusting the feature extraction parameter `min_distance` in `src/gnn_inference/segmentation/segment_graph.py` accordingly.




## **GNN Training Recipe**
The following instructions will help you configure parameters to generate synthetic layout data, augment the Sanskrit dataset, prepare data for GNN training, and train a custom GNN architectures to perfrom text-line segmentation, which is formulated as an edge classification task.


#### 🔵 Install Conda Environment
```bash
cd src
conda env create -f environment.yaml
conda activate gnn_layout
```

#### 🔵 To Flatten Sanskrit Dataset (optional)
The flattened sanskrit dataset is already provided in `src/gnn_data/flattened_sanskrit_data/` (without original images). However, if you wish to flatten the original hierarchical dataset again (with the original images) with custom feature engineering, you can run:

```bash
cd dataset

python flatten.py
```
This will create a new folder `src/gnn_data/flattened_sanskrit_data/` with all the data files flattened into a single directory structure, with an `index.csv`.

#### 🔵 Generate Synthetic Data
Configure the parameters in `src/configs/synthetic.yaml` as needed, then run:
```bash
cd src

python synthetic_data_gen/generate.py --dry-run --config configs/synthetic.yaml  # to visualize a few samples
python synthetic_data_gen/generate.py --config configs/synthetic.yaml
```

This will create a new folder `src/gnn_data/synthetic_layout_data/` with all the generated synthetic data files in the graph based format.

This script peforms domain randomization to generate synthetic layout data simulating complex layouts in the graph based formulation introduced in this project. Both the synthetic data and the real data use the same graph based format, making it easy to integrate synthetic data into training pipelines.

#### 🔵 To Augment Sanskrit Dataset
Configure the parameters in `src/configs/augment.yaml` as needed, then run:
```bash
cd src

python synthetic_data_gen/augment.py \
--config configs/augment.yaml \
--input_dir "gnn_data/flattened_sanskrit_data/gnn-dataset" \
--output_dir "gnn_data/augmented_sanskrit_dataset/"
```
This will create a new folder `src/gnn_data/augmented_sanskrit_dataset/` with three subfolders: `train`, `val` and `test`. `train` will contain the augmented training samples, while `val` and `test` will contain the original validation and test samples respectively.


#### 🔵 Create Combined Dataset (Synthetic + Augmented Real Data)
First, copy synthetic data, augmented sanskrit data (training set) into a single folder. For example, you can create a new folder `src/gnn_data/combined_data/` and copy the following into it:
```bash
cd src

mkdir -p gnn_data/combined_data/

rsync -a gnn_data/generated_synthetic_data/ gnn_data/combined_data/
rsync -a gnn_data/augmented_sanskrit_dataset/train/ gnn_data/combined_data/
echo "augmented real data + synthetic data prepared at: gnn_data/combined_data/"
```

Hence our training dataset will be at `src/gnn_data/combined_data/`
validation dataset at `src/gnn_data/augmented_sanskrit_dataset/val/` 
and test dataset at `src/gnn_data/augmented_sanskrit_dataset/test/` (unused as of now).

#### 🔵 Prepare Data for GNN Training
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

#### 🔵 Train GNN Model
First configure the GNN training parameters in `configs/gnn_training.yaml` as needed, then run:
```bash
cd src

python -m gnn_training.training.main_train_eval \
--config "configs/gnn_training.yaml" \
--dataset_path "gnn_data/processed_data_gnn/" \
--unique_folder_name "gnn_experiment_1" \
--gpu_id 1
```
This will create a new folder `src/gnn_training/training_runs/gnn_experiment_1/`.


## Acknowledgements
We would like to thank Petar Veličković, Oliver Hellwig, Dhavel Patel for their extermely valuable inputs and discussions.


## **TODO List**
*   [x] Generete larger and diverse synthetic layout dataset
*   [x] Perform multi-task GNN training: Text Box detection, reading order prediction along with line segmentation
*   [x] Integrate GNN model into the manuscript layout analysis tool. Link: [Manuscript Annotation Tool](https://github.com/flame-cai/win64-local-ocr-tool/tree/GNN-DEV-MAIN). 