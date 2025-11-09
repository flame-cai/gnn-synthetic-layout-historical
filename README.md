# Towards text-line Segmentation from Sanskrit Manuscripts using Graph Neural Networks

**Version:** 1.0
**Last Updated:** November 9, 2025

## Dataset Overview

Each top-level folder represents a manuscript. Inside, the data is organized as follows:

###### 📁 `images/` Contains the original, high-resolution scanned images of manuscript pages.
*   `[PAGE_ID].jpg`: Original images are not provided with the dataset due to copyright reasons, however they can be accessed online at the URLs provided

###### 📁 `page-xml-graph-groundtruth/` Contains ground truth annotations in the form of bounding polygons.
*   `[PAGE_ID].xml`

###### 📁 `page-xml-rectangle/` Contains non-overlapping ground truth annotations in the form of bounding polygons.
*   `[PAGE_ID].xml`

###### 📁 `heatmaps/` Contains the output of CRAFT, which is a heatmap (with resolution downscaled by 2 maintaining aspect ratio)
*   `[PAGE_ID].jpg`

###### 📁 `gnn-dataset/` Contains pre-processed, model-ready data for training Graph Neural Networks.
*   `[PAGE_ID]_dims.txt`: Dimensions of heatmap (downscaled by 2 compared to the original image)
*   `[PAGE_ID]_inputs_normalized.txt`: Normalized node features for the GNN, maintaining aspect ratio
*   `[PAGE_ID]_inputs_unnormalized.txt`: Unnormalized node features (downscaled by 2 compared to the original image)
*   `[PAGE_ID]_labels_region.txt`: Ground truth labels for region classification. All points belonging to the same text-box share the same label
*   `[PAGE_ID]_labels_textline.txt`: Ground truth labels.  All points belonging to the same text-line share the same label

###### 📄 `index.csv` (Master Index File) This file acts as the master index for the entire dataset, providing metadata for each page.
*   `short_id`: A unique, zero-padded 6-digit identifier for each page entry.
*   `original_unique_id`: The primary file identifier (`[PAGE_ID]`) used across all subdirectories.
*   `dataset`: The name of the parent dataset collection (e.g., `sanskrit-manuscripts`).
*   `sub_manuscript_id`: The human-readable name of the manuscript folder (e.g., `ravisankrantivicharah`).
*   `layout`: A label indicating the complexity of the page layout (`simple` or `complex`).


## Source code
To be updated soon!!