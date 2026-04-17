# EVAL.md

This document is the evaluation blueprint for the repository. It covers both the current regression gate and the slower OCR active-learning research harness. It should be treated as the source of truth when changing evaluation thresholds, adding datasets, or extending the fine-tuning workflow.

The short version is this: evaluation in this repository is not just about model accuracy on a fixed benchmark. The long-term goal is to reduce the total human effort required to digitize a manuscript page by page while preserving or improving output quality.

## Why Evaluation Exists Here

The repository has two connected products:

1. The GNN layout analysis core in `src/`
2. The semi-automatic annotation and recognition tool in `app/`

The app already lets a human correct:

- character-detection failures by adding and deleting nodes
- graph mistakes by adding and deleting edges
- text-region grouping by assigning textbox labels
- OCR mistakes by editing text

That means evaluation here has to answer two different questions:

1. Does the automatic pipeline still work after a code change?
2. Are we reducing manual effort over time as corrections are fed back into model updates?

The first question is covered by the existing headless regression gate. The second question is the main active-learning direction.

## Current Source Of Truth

The current evaluation code lives in:

- `app/tests/test_ci_e2e.py`
- `app/tests/test_recognition_active_learning_unit.py`
- `app/tests/test_recognition_finetuning_e2e.py`
- `app/tests/evaluate.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/recognition/active_learning.py`
- `app/recognition/pagexml_line_dataset.py`
- `app/tests/eval_dataset/`
- `app/tests/logs/`
- `.githooks/pre-commit`
- `scripts/run_precommit_eval.py`

## Evaluation Layers

### Level 0: Fast structural checks

Purpose: catch obvious breakage cheaply.

Examples:

- PAGE-XML parsing checks
- OCR selector unit tests
- padding-policy tests
- oversampling and augmentation tests

These checks are necessary but not sufficient. They prove that narrow mechanics still behave correctly.

### Level 1: Automatic end-to-end regression

Purpose: verify that the full automatic path still works on a fixed dataset.

Current entrypoint:

- `app/tests/test_ci_e2e.py`

This gate runs:

1. manuscript upload
2. CRAFT plus GNN inference
3. PAGE-XML generation
4. OCR
5. evaluation against PAGE-XML ground truth

This remains the required pre-commit quality gate for changes that can affect the app, OCR, PAGE-XML generation, evaluation code, or GNN inference.

### Level 2: Offline OCR active-learning research harness

Purpose: evaluate whether sequential OCR fine-tuning improves later pages in a reproducible way.

Current entrypoint:

- `app/tests/test_recognition_finetuning_e2e.py`

This harness does not use CRAFT or GNN segmentation. Instead it:

1. prepares perfect line crops from PAGE-XML ground truth
2. fine-tunes the local OCR checkpoint sequentially on earlier pages
3. evaluates later pages after each update
4. writes full run artifacts under `app/tests/logs/`

This harness is offline-first and research-oriented. It is not yet the GUI fine-tuning loop.

### Level 3: Future GUI and human-in-the-loop evaluation

Purpose: measure whether real user correction effort drops as the system learns.

This layer is still future work. It will require structured logging from the UI and the backend, not just model outputs.

## Current OCR Harness State

As of 2026-04-17, the OCR active-learning harness has the following implemented features:

- CER-aligned sibling checkpoint selection between `best_accuracy.pth` and `best_norm_ED.pth`
- explicit width policies: `global_2000_pad` and `batch_max_pad`
- bounded CER-weighted oversampling
- OCR-only augmentation policies
- LR scheduler plumbing
- richer artifacts including `curve_metrics.json`, `per_page.csv`, `per_line.csv`, `selector_metrics.json`, `fine_tune_metadata.json`, and plots
- focused unit coverage for selector choice, width policy, oversampling, and augmentation

Relevant files:

- `app/recognition/active_learning.py`
- `app/recognition/dataset.py`
- `app/recognition/ocr_defaults.py`
- `app/recognition/train.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`

## Latest Completed OCR Study

The latest completed study is:

- `app/tests/logs/20260417_155737_ocrft_eval_dataset/`

Dataset setup in that run:

- ordered pages: `233_0002` through `233_0016`
- sequential fine-tune pages: `233_0002` through `233_0006`
- evaluation pages: `233_0011` through `233_0016`
- batch size: `1`
- optimizer: `Adadelta`

Primary metric:

- `early_weighted_page_cer`
- formula: `sum((K - s + 1) * page_CER_s) / sum(K - s + 1)` over steps `s = 0..K`
- in the 5-page study, `K = 5`

Regression guard:

- no step may worsen aggregate page CER by more than `0.005` absolute versus the previous step

Important results from the 2026-04-17 study:

- Primary-metric winner: `wb_oc_an_sn020`
  Meaning: `width_policy=batch_max_pad`, `oversampling_policy=cer_weighted`, `augmentation_policy=none`, `lr_scheduler=none`, `lr=0.2`
  Evidence: `curve_metric_value=0.25063928319742274`
- Best `final_page_cer`: `wb_on_an_sn020`
  Evidence: `final_page_cer=0.18266384778012684`
- Best `first_step_gain`: `wb_on_an_sn020`
  Evidence: `first_step_gain=0.05433403805496828`

The canonical human-readable summary files are:

- `app/tests/logs/recognition_finetune_results_latest.md`
- `app/tests/logs/recognition_finetune_results_latest.json`

## Required Artifacts

Serious evaluation runs should write a dedicated artifact folder under `app/tests/logs/`.

The current OCR study shape is:

    app/tests/logs/<timestamp>_ocrft_<dataset>/
      config.json
      summary.md
      metrics.json
      gt_eval_subset/
      prepared_pages/
      policies/
        <policy_slug>/
          config.json
          summary.md
          metrics.json
          curve_metrics.json
          per_page.csv
          per_line.csv
          selector_metrics.json
          fine_tune_metadata.json
          predicted_page_xml/
          plots/

Interpretation:

- `config.json`: dataset setup, device info, and policy values
- `summary.md`: readable summary of the run
- `metrics.json`: machine-readable aggregate metrics
- `curve_metrics.json`: primary curve metric and regression-guard outcome
- `per_page.csv`: page-level metrics for each fine-tune step
- `per_line.csv`: per-line OCR detail including width and padding metadata
- `selector_metrics.json`: sibling checkpoint ranking evidence
- `fine_tune_metadata.json`: training options, selected checkpoint, and timing

This OCR artifact layout should be preserved unless there is a strong reason to change it.

## Current Commands

### Fast regression gate

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

### OCR unit tests

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_unit -v

### Slow OCR study verifier

On Windows, prefer an activated `gnn_layout` shell or the direct environment Python for this command. Very long Unicode-heavy output can trigger a `conda run` printing failure even after the study itself has finished and written artifacts.

Preferred forms:

    conda activate gnn_layout
    python -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

or:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

If the wrapper crashes after artifact creation, inspect the newest timestamped folder under `app/tests/logs/` before assuming the study itself failed.

## Metrics By Task

### 1. Character detection or a future surrogate CRAFT task

What matters:

- missing character centers
- false character centers
- downstream effect on text-line segmentation
- amount of manual node correction required

Current reality:

- the app logs node corrections in `app/app.py`
- there is no formal node-level benchmark yet

### 2. GNN edge classification for text-line segmentation

What matters:

- PAGE-level CER
- line-level CER at polygon IoU thresholds
- how easy graph mistakes are to repair

Current reality:

- automatic evaluation already computes PAGE-level and line-level CER through `app/tests/evaluate.py`
- edge-edit telemetry is not yet a first-class evaluation artifact

### 3. Text-region grouping

What matters:

- correct region assignment
- how often users need to merge or split regions manually

Current reality:

- the UI supports region grouping
- there is no formal region-grouping evaluation dataset yet

### 4. Text recognition

What matters:

- OCR CER
- confidence summaries
- first-step gain
- final-page CER
- early-weighted curve quality across sequential fine-tuning steps
- eventually, human text-correction effort

Current reality:

- automatic OCR evaluation exists
- an offline sequential OCR fine-tuning harness exists
- the GUI still does not promote fine-tuned OCR checkpoints

## Human Effort Metrics

Because the long-term goal is active learning, human effort must eventually become a first-class metric.

Future human-in-the-loop evaluation should log, per page:

- total correction time
- nodes added
- nodes deleted
- edges added
- edges deleted
- textbox edits
- OCR text edits
- save cycles
- fine-tune triggers
- active checkpoint id and candidate checkpoint id

## Current Evaluation Limits

The repository has meaningful OCR research infrastructure now, but the current benchmark is still narrow.

Current limits:

- only one OCR evaluation dataset is wired into the harness
- the latest completed broad search used 5 fine-tune pages
- there is still no GUI-triggered fine-tuning
- there is still no formal model registry or promotion rule
- human correction effort is not yet logged as structured data

## Near-Term Recommended Additions

The next high-value steps are:

1. Run a focused 9-page OCR follow-up study comparing only `wb_oc_ar_sn020`, `wb_oc_an_sn020`, and `wb_on_an_sn020` across learning rates `{0.01, 0.2, 0.8}`.
2. Harden the Windows OCR-study execution path so long test output is captured safely without the `conda run` Unicode printing failure.
3. Add a small OCR model registry with active, candidate, and rollback metadata.
4. Add structured OCR text-edit and fine-tune-trigger logging from the save and recognition flows in `app/app.py`.
5. Add GUI-safe background job orchestration so training and inference do not contend for the same device.
6. Extend the OCR benchmark to multiple manuscript sequences after the focused 9-page study stabilizes.

## Definition Of Success

Evaluation maturity in this repository means:

- every significant code change can be screened by an automatic headless gate
- every OCR research claim can be tied to a saved artifact folder and exact settings
- the OCR verifier can rank policies by the repository's chosen primary metric
- GUI fine-tuning, when added, promotes models only through a recorded verifier-backed rule
- human correction burden is eventually measured directly rather than inferred indirectly
