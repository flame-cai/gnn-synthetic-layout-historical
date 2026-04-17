# Recognition Fine-Tuning Session Report

Date: 2026-04-16  
Environment: `C:\Users\intro\miniconda3\envs\gnn_layout\python.exe`  
Repository: `gnn-synthetic-layout-historical`

## Goal

Implement a non-GUI OCR fine-tuning path for the local recognition checkpoint `app/recognition/pretrained_model/vadakautuhala.pth`, driven by PAGE-XML ground truth from `app/tests/eval_dataset`, and validate it with a sequential active-learning-style experiment.

## What Was Implemented

### 1. OCR package cleanup and shared defaults

The OCR code under `app/recognition/` was made usable as a real package instead of only through Flask path hacks.

Implemented:

- Shared OCR defaults and checkpoint-loading helpers in [app/recognition/ocr_defaults.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/ocr_defaults.py:1)
- Package-safe imports in:
  - [app/recognition/train.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/train.py:1)
  - [app/recognition/test.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/test.py:1)
  - [app/recognition/model.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/model.py:1)
  - [app/recognition/recognize_manuscript_text_v2_pretrained.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/recognize_manuscript_text_v2_pretrained.py:1)
- Windows-safe dataset handling fixes in [app/recognition/dataset.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/dataset.py:1)

Important fix:

- The pretrained checkpoint loader now correctly handles `module.*` keys from `vadakautuhala.pth`. The original fine-tune path would have silently mismatched DataParallel keys.

### 2. PAGE-XML driven ground-truth line extraction

Implemented a reusable GT crop-preparation module in [app/recognition/pagexml_line_dataset.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/pagexml_line_dataset.py:1).

Capabilities:

- Parses PAGE-XML `TextRegion` and `TextLine` entries
- Preserves `textbox_label_*` and `structure_line_id_*` custom identifiers already present in the GT XML
- Sorts lines with the same `(y_center, x_min)` logic used by [app/tests/evaluate.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/evaluate.py:1)
- Crops grayscale polygon-masked line images using the same median-background masking style as the app export path
- Writes two outputs per page:
  - app-style crops under `image-format/<page>/textbox_label_*/line_*.jpg`
  - fine-tuning data under `finetune_dataset/test/*.png` plus `gt.txt`
- Writes a manifest so each flattened line image can be traced back to page, region, line id, and polygon

### 3. Fine-tuning orchestration

Implemented the fine-tuning pipeline in [app/recognition/active_learning.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/active_learning.py:1).

Capabilities:

- Prepares page datasets from GT PAGE-XML
- Builds replay corpora for sequential fine-tuning
- Converts training corpora into LMDB
- Fine-tunes from a specified base checkpoint with batch size `1`
- Versions each run under a dedicated output directory
- Generates PAGE-XML predictions from a chosen checkpoint
- Persists fine-tune metadata, including:
  - checkpoint lineage
  - page ids used in the corpus
  - train/validation sample counts
  - training options
  - train duration

Important design changes made during the session:

- Added cumulative replay support so each step can train on all corrected pages seen so far
- Added deterministic split infrastructure, though the current evaluator default is `validation_ratio=0.0`
- Added dataset manifests and metadata needed for experiment reproducibility

### 4. LMDB tooling

Added [app/recognition/lmdb_tools.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/lmdb_tools.py:1).

Important fix:

- The Windows LMDB `map_size` is now estimated dynamically instead of using an impractically large static allocation that failed with local disk-space errors.

### 5. Sequential evaluator and test harness

Implemented:

- Dataset config registry in [app/tests/recognition_finetuning_config.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/recognition_finetuning_config.py:1)
- Experiment runner in [app/tests/recognition_finetuning_experiment.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/recognition_finetuning_experiment.py:1)
- Slow end-to-end test in [app/tests/test_recognition_finetuning_e2e.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/test_recognition_finetuning_e2e.py:1)

The evaluator writes timestamped artifacts under `app/tests/logs/`, including:

- `config.json`
- `summary.md`
- `metrics.json`
- `per_page.csv`
- `fine_tune_metadata.json`
- predicted PAGE-XMLs
- versioned model checkpoints
- CER-vs-pages plot

The evaluator was also fixed so that a failing step is still recorded in the run artifacts before the assertion aborts the test.

## Experiment History

Full experiment runs completed during this session:

1. `20260416_210927_recognition_finetune_eval_eval_dataset`
   Failure: step 4 regression with page-only sequential fine-tuning and the original aggressive schedule

2. `20260416_215952_recognition_finetune_eval_eval_dataset`
   Failure: step 1 regression after enabling a 20% validation split on very small data

3. `20260416_220541_recognition_finetune_eval_eval_dataset`
   Failure: step 2 regression under cumulative replay with `lr=0.8`, `num_iter=100`

4. `20260416_222641_recognition_finetune_eval_eval_dataset`
   Failure: step 4 regression under cumulative replay with the gentler schedule `lr=0.2`, `num_iter=60`

The failure log has been updated in [docs/exec-plans/active/recognition-finetuning-failure-log.md](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/docs/exec-plans/active/recognition-finetuning-failure-log.md:1).

## Current Best-Known Experiment Configuration

The strongest configuration found in this session is:

- training policy: cumulative replay
- validation ratio: `0.0`
- optimizer: `Adadelta`
- learning rate: `0.2`
- iterations: `60`
- validation interval: `5`
- batch size: `1`
- workers: `0`

This is now the default in [app/tests/recognition_finetuning_config.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/recognition_finetuning_config.py:1).

## Latest Full-Run Results

Latest run:

- [app/tests/logs/20260416_222641_recognition_finetune_eval_eval_dataset](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/logs/20260416_222641_recognition_finetune_eval_eval_dataset)

Selected-checkpoint CER curve from the latest recorded `metrics.json`:

- Step 0 pretrained: `0.309302`
- Step 1 selected checkpoint: `0.308457`
- Step 2 selected checkpoint: `0.263002`
- Step 3 selected checkpoint: `0.220296`
- Step 4 selected checkpoint: `0.227907`

That run failed because step 4 regressed relative to step 3.

## Most Important Finding

The remaining blocker is now checkpoint selection, not data preparation or fine-tune orchestration.

Evidence from the latest run:

- Step 4 selected checkpoint was `best_norm_ED.pth` and produced page CER `0.227907`
- The sibling step-4 checkpoint `best_accuracy.pth` produced page CER `0.216913`
- Probing `best_accuracy.pth` across the latest run gives:
  - Step 1: `0.306765`
  - Step 2: `0.276321`
  - Step 3: `0.220296`
  - Step 4: `0.216913`

That means the training policy is close enough for the experiment to work, but the post-training checkpoint selector is still choosing the wrong saved model on at least some steps.

## Recommended Next Action

The next session should change the checkpoint selector in [app/recognition/active_learning.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/active_learning.py:1).

Recommended approach:

1. After training, evaluate both `best_accuracy.pth` and `best_norm_ED.pth` on the fine-tune validation corpus.
2. Compute actual character error rate on that corpus.
3. Select the checkpoint with lower CER instead of hard-preferring `best_norm_ED.pth`.
4. Rerun the full experiment from step 0 through step 5.

Why this is the right next step:

- It avoids using the evaluation pages as the selector.
- It uses the same target metric family the experiment cares about.
- The latest probes strongly suggest that this change is sufficient to clear the step-4 blocker.

## Verification Performed

During the session I successfully verified:

- syntax/import smoke for the modified OCR modules
- GT PAGE-XML preparation on the evaluation dataset
- OCR baseline inference from the pretrained checkpoint
- fine-tune training runs under `gnn_layout`
- repeated full sequential evaluator runs, producing durable artifacts and failure evidence

## Final Status

The fine-tuning code path and the GT-driven evaluator are implemented and working end to end, but the acceptance test does not yet pass because the checkpoint selector still prefers the wrong saved model on later sequential steps.
