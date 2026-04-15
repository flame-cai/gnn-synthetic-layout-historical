# AGENTS.md

This file is for coding agents working in this repository. It explains how the repository works, maps the respective research paper to the codebase, and gives a practical guide for setting up and using the project in automatic and semi-automatic modes.

# This repository has two parts:

##  💻 **Graph Neural Network based Text-Line Segmentation Core ```src/```**

'src/' : This directory contain implementation of the paper:
- `chincholikar26towards.pdf`
- Title: `Towards Text-Line Segmentation of Historical Documents Using Graph Neural Networks`
- Authors: `Kartik Chincholikar`, `Kaushik Gopalan`, and `Mihir Hasabnis`
- Venue: `ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling`
The paper presents a graph-based problem formulation for performing text-line segmentation of historical documents, by representing characters (or grapheme clusters) as the nodes, and with edges connecting characters to their previous and next characters on the text-line. This converts the image segmentation learning task into a binary edge classification learning task for Graph Neural Networks. This also enables training on large-scale synthetic data simulating complex layouts, enabling better robustness to Layout-level distribution shifts observed in historical documents.

Pipeline: 
1. Use CRAFT to detect character regions from the historical manuscript image and produce a heatmap.
2. Convert the heatmap to a point cloud of character centers and radii. (We can also use the synthetic data generator to directly get synthetic point clouds of complicated layouts. This skips step 1)
3. Build a heuristic graph using geometric priors:
   - characters on a line should usually have two opposite neighbors
   - character spacing is usually smaller than line spacing
4. Add extra candidate connectivity so true text-line edges are available to the model.
5. Build node and edge features from geometry plus heuristic metadata.
6. Train a GNN to classify candidate edges as keep/delete.
7. Convert kept edges into connected components, which, along with the CRAFT heatmap are used to segment the final textlines.
8. Export predictions as graph labels, PAGE-XML, and cropped line images for text-line OCR.

'src/gnn_inference' contains code for end-to-end inference using the pipeline using pre-trained CRAFT and GNN.
'src/synthetic_data_gen' contains code for generating synthetic layout data, which can be configured in 'src/configs/synthetic.yaml'
'src/synthetic_data_gen' also contains code for augmenting real layout data, which can be configured in 'src/configs/augment.yaml'
'src/gnn_training/gnn_data_preparation'  contains code for preprocessing the synthetic data and augmented real data, which can be configured in 'src/configs/gnn_preprocessing.yaml'
'src/gnn_training/training'  contains code for the training various GNN architectures, which can be configured in 'src/configs/gnn_training.yaml'

## 🧩 **Semi Automatic Annotation Tool ```app/```**

'app/' : This directory contains an application which has the graph-based text-line segmentation at it core, but it has additional features:
- It allows for semi-automatic text-line segmentation, by allowing users to add or delete nodes in cases of CRAFT failures. It also allows users to add or delete edges when the GNN incorrectly does the binary edge classification. The also allows allows manually grouping all text-lines belonging to the same text-region. 

- Once the text-lines are segmented, the app enables recognizing the text-content from the segmented text-lines using two methods: a) EasyOCR based recognition b) Gemini based recognition. In other words, the app has two modes. In the layout analysis mode, the text-lines are segmented and the PAGE-XML files are created, and in the Recognition Mode, the text content from the text-lines is recognized, and the PAGE-XML files are updated with the unicode text. 



# Documentation References:

'./README.md': This section provides additional introduction and outlines how to setup and install a) the Graph Neural Network based Text-Line Segmentation Core and b) The Semi Automatic Annotation Tool. Please run tests and verifications using the conda environment gnn_layout. If you cannot find the conda environment 'gnn_layout', please install it as mentioned in README.md

'./ENGINEERING_DOCTRINE.md': This file describes how Agents working on the code base should behave like.

'./VISION.md': This section describes the Vison of the direction we want to slowly develop this repository

'./PLANS.md': When writing complex features or significant refactors, use an ExecPlan from design to implementation. Use `PLANS.md` for ExecPlan structure and standards. Use only `docs/exec-plans/active/`, `docs/exec-plans/completed/`, and `docs/exec-plans/proposed/`. Keep proposed plans rollout-ordered and blocker-first.

'./docs/exec-plans/tech-debt-tracker.md': current short list of the highest-priority debts






