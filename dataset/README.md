
## Each top-level folder represents a manuscript📜. Inside, the data is organized as follows:

#### 📁 `images/` 
Contains the original, high-resolution scanned images of manuscript pages.
*   `[PAGE_ID].jpg`: Original images are not provided with the dataset due to copyright reasons, however they can be accessed online via the URLs provided in the `manuscript_urls.txt` file provided.

#### 📁 `page-xml-graph-groundtruth/` 
Contains ground truth annotations in the form of bounding polygons in the standard PAGE XML format commonly used for historical document analysis.
*   `[PAGE_ID].xml`

#### 📁 `page-xml-rectangle/` 
Contains non-overlapping ground truth annotations in the form of bounding polygons. in the standard PAGE XML format commonly used for historical document analysis.
*   `[PAGE_ID].xml`

#### 📁 `heatmaps/` 
Contains the output of CRAFT, which is a heatmap (with resolution downscaled by 2 maintaining aspect ratio)
*   `[PAGE_ID].jpg`

#### 📁 `gnn-dataset/` 
Contains pre-processed, model-ready data for training Graph Neural Networks.
*   `[PAGE_ID]_dims.txt`: Dimensions of heatmap (downscaled by 2 compared to the original image)
*   `[PAGE_ID]_inputs_normalized.txt`: Normalized node features for the GNN, maintaining aspect ratio (x,y,pseudo_fontsize). pseudo_fontsize is calculated based on the heatmap.
*   `[PAGE_ID]_inputs_unnormalized.txt`: Unnormalized node features (downscaled by 2 compared to the original image)
*   `[PAGE_ID]_labels_region.txt`: Ground truth labels for region classification. All points belonging to the same text-box share the same label
*   `[PAGE_ID]_labels_textline.txt`: Ground truth labels.  All points belonging to the same text-line share the same label

## To Flatten the Dataset
```bash
cd dataset
python flatten.py
```


This will create a new folder `src/gnn_data/flattened_sanskrit_dataset/` with all the data files flattened into a single directory structure, with an `index.csv` file at the root level.

This file acts as the master index for the entire dataset, providing metadata for each page.
*   `short_id`: A unique, zero-padded 6-digit identifier for each page entry.
*   `original_unique_id`: The primary file identifier (`[PAGE_ID]`) used across all subdirectories.
*   `dataset`:  The manuscript name a page belongs to (e.g., `ravisankrantivicharah`).
*   `layout`: A label indicating the complexity of the page layout (`simple` or `complex`).
