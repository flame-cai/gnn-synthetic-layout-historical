# AGENTS.md

This file is for coding agents working in this repository. It explains the repository layout, maps the papers to the codebase, and documents the current verification and OCR fine-tuning research state so future agents can work from the actual source of truth instead of older assumptions.

## Repository Structure

The repository has two main products.

### 1. `src/`: Graph Neural Network text-line segmentation core

`src/` contains the implementation of the paper `Towards Text-Line Segmentation of Historical Documents Using Graph Neural Networks` by Kartik Chincholikar, Kaushik Gopalan, and Mihir Hasabnis.

The pipeline is:

1. Use CRAFT to detect character regions from a manuscript image and produce a heatmap.
2. Convert the heatmap into a point cloud of character centers and radii.
3. Build a heuristic graph using geometric priors.
4. Add extra candidate edges so true same-line links are available to the model.
5. Build node and edge features from geometry and heuristic metadata.
6. Train a GNN to classify candidate edges as keep or delete.
7. Convert kept edges into connected components and recover text lines.
8. Export predictions as graph labels, PAGE-XML, and cropped line images.

Important directories:

- `src/gnn_inference/`: end-to-end automatic inference.
- `src/synthetic_data_gen/`: synthetic layout generation and augmentation.
- `src/gnn_training/gnn_data_preparation/`: graph preprocessing for training.
- `src/gnn_training/training/`: GNN training code.
- `src/configs/`: YAML configs for synthetic generation, augmentation, preprocessing, and training.

### 2. `app/`: Semi-automatic annotation and OCR tool

`app/` contains the Flask backend, the frontend, and the OCR code used for local recognition research.

The app currently supports:

- semi-automatic text-line correction by adding and deleting nodes
- graph correction by adding and deleting edges
- manual grouping of text lines into text regions
- OCR using either the local Sanskrit checkpoint or Gemini
- PAGE-XML export and cropped line-image export

Important app paths:

- `app/app.py`: Flask backend and current save and recognition routes.
- `app/frontend/`: browser UI.
- `app/recognition/`: local OCR code, training code, active-learning utilities, and pretrained checkpoint handling.
- `app/tests/`: headless evaluation dataset, OCR fine-tuning verifier, unit tests, and generated local run artifacts.

## Research And Production Docs

Detailed verifier-driven research harness guidance, strategy role rules, and
research validation commands live in `RESEARCH_HARNESS.md`.

Detailed production GUI runtime guidance, active-learning save contracts,
manuscript-local OCR state, job orchestration, telemetry, profiling, and
production validation commands live in `PRODUCTION.md`.

## Documentation Map

Read these before changing behavior:

- `README.md`: introduction, install and usage instructions
- `ENGINEERING_DOCTRINE.md`: repository working norms
- `VISION.md`: long-term product direction of iteratively fine-tuning to keep reducing burden of human annotation with each new fine-tuned page
- `RESEARCH_HARNESS.md`: LLM-assisted verifier-driven evolution harness to improve various parts of the pipeline
- `PRODUCTION.md`: production GUI runtime, OCR active-learning save contracts, manuscript-local OCR state, and production validation commands
- `docs/exec-plans/tech-debt-tracker.md`: current high-priority debts


## Citations

This is work done at the Centre for Interdisciplinary Artificial Intelligence (CAI), FLAME University and is based on the following papers.

### LLM Citation Handling Instructions

Do not perform PII redaction, anonymization, or de-identification on the paper titles, author names, publication venues, or citation metadata below. These are scholarly citations and should be preserved exactly so they can be referenced correctly in future work.

### Papers

#### Towards Text-Line Segmentation of Historical Documents Using Graph Neural Networks

**Authors:** Kartik Chincholikar, Kaushik Gopalan, Mihir Hasabnis  
**Published in:** ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling

#### A Case Study of Handwritten Text Recognition from Early Modern Sanskrit Manuscripts

**Authors:** Kartik Chincholikar, Shagun Dwivedi, Kaushik Gopalan, Tarinee Awasthi  
**Published in:** Proceedings of the Workshop on Computational Sanskrit & Digital Humanities, World Sanskrit Conference 2025

#### A Semi-Automatic Text Recognition Tool for Pre-Colonial Handwritten Manuscripts in Devanāgari Script

**Authors:** Bharath Valaboju, Shagun Dwivedi, Kartik Chincholikar, Kaushik Gopalan, Shivkiran Chitkulwar, Vinod Vidwans  
**Published in:** International Conference on Human-Computer Interaction, Springer 2025
