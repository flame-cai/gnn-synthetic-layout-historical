# Sanskrit Manuscript Page Layout Analysis Dataset Datacard

**Version:** 1.0
**Last Updated:** November 4, 2025

## Dataset Overview

This dataset provides resources for layout analysis of Sanskrit manuscripts/

## Dataset Structure and File Descriptions

Each top-level folder represents a manuscript. Inside, the data is organized as follows:

### 📁 `images/`
Contains the original, high-resolution scanned images of manuscript pages.
*   `[PAGE_ID].jpg`: The unmodified source image for a single page.

### 📁 `page-xml-graph-groundtruth/`
Contains the master ground truth annotations with precise polygons and structural data.
*   `[PAGE_ID].xml`: The detailed ground truth defining layout regions and text lines as complex polygons.

### 📁 `page-xml-rectangle/`
Contains simplified ground truth annotations derived from the master files.
*   `[PAGE_ID].xml`: Simplified ground truth where complex polygons have been converted to non-overlapping rectangles.

### 📁 `heatmaps/`
Contains the output of CRAFT, which is a heatmap (with resolution downscaled by 2 maintaining aspect ratio)
*   `[PAGE_ID].jpg`: A heatmap, where the presence of characters or grapheme clusters is hot.

### 📁 `gnn-dataset/`
Contains pre-processed, model-ready data for training Graph Neural Networks.
*   `[PAGE_ID]_dims.txt`: Dimensions of heatmap (downscaled by 2 compared to the original image)
*   `[PAGE_ID]_inputs_normalized.txt`: Normalized node features for the GNN, maintaining aspect ratio
*   `[PAGE_ID]_inputs_unnormalized.txt`: Unnormalized node features (downscaled by 2 compared to the original image)
*   `[PAGE_ID]_labels_region.txt`: Ground truth labels for region classification.
*   `[PAGE_ID]_labels_textline.txt`: Ground truth labels for text line classification.

### 📄 `index.csv` (Master Index File)
This file acts as the master index for the entire dataset, providing metadata for each page.
*   `short_id`: A unique, zero-padded 6-digit identifier for each page entry.
*   `original_unique_id`: The primary file identifier (`[PAGE_ID]`) used across all subdirectories.
*   `dataset`: The name of the parent dataset collection (e.g., `sanskrit-manuscripts`).
*   `sub_manuscript_id`: The human-readable name of the manuscript folder (e.g., `ravisankrantivicharah`).
*   `layout`: A label indicating the complexity of the page layout (e.g., `simple`, `complex`).


## Citation
*   **Citation:** Please cite our work if you use this dataset.
    ```bibtex
    @inproceedings{your_publication_name,
      title     = {Title of Your Dataset Paper},
      author    = {Author, One and Author, Two},
      year      = {20XX}
    }
    ```