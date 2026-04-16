# EVAL.md

This document is the blueprint for evaluation in this repository. It is meant to guide both the current automatic checks and the future human-in-the-loop and active-learning evaluations described in [VISION.md](./VISION.md).

The short version is this: evaluation in this codebase is not only about model accuracy on a fixed dataset. The long-term goal is to reduce total human effort required to digitize a manuscript page-by-page while preserving or improving output quality. That means future evaluation must measure both output quality and correction effort across repeated fine-tuning cycles.

## Why Evaluation Exists Here

This repository has two connected products:

1. The GNN layout analysis core in `src/`
2. The semi-automatic annotation and recognition tool in `app/`

The current tool already supports human correction of:

- character detection failures by adding or deleting nodes
- text-line graph errors by adding or deleting edges
- text-region grouping by assigning textbox labels
- OCR errors by editing recognized text

That makes the repository naturally suited to active learning. The evaluation system therefore needs to answer two different questions:

1. Does the automatic pipeline still work after a code change?
2. Are we reducing manual effort over time as corrections are fed back into fine-tuning loops?

The first question is covered today by an automatic pre-commit test. The second question is the main future direction.

## Current Source Of Truth

The current automatic evaluation flow lives in these files:

- `app/tests/test_ci_e2e.py`
- `app/tests/evaluate.py`
- `app/tests/eval_dataset/images/`
- `app/tests/eval_dataset/labels/PAGE-XML/`
- `app/tests/logs/`
- `.githooks/pre-commit`
- `scripts/run_precommit_eval.py`

The current pre-commit gate runs the following workflow without the GUI:

1. Upload the 15-page evaluation manuscript.
2. Run CRAFT plus GNN inference.
3. Save PAGE-XML for each page.
4. Run local OCR on each page.
5. Compare predicted PAGE-XML against ground-truth PAGE-XML.
6. Fail the commit if aggregate quality falls below the defined thresholds.

This is a regression gate. Its purpose is to catch breakage in the existing prototype path before code is committed.

## Evaluation Principles

All evaluation added to this repository should follow these principles:

- Prefer end-to-end behavior over isolated metric theater.
- Measure the user-visible output, not only internal tensors.
- Keep every evaluation reproducible: same inputs, same settings, same logged artifacts.
- Separate automatic regression gates from slower exploratory experiments.
- When human interaction is involved, log the human effort explicitly.
- When active learning is involved, evaluate improvement page-by-page, not only before-vs-after on a static test split.
- Store enough metadata to explain why a run passed, failed, improved, or regressed.

## Evaluation Ladder

Evaluation should grow in layers. Not every change needs every layer, but the layers define the target architecture for the repository.

### Level 0: Fast Structural Checks

Purpose: catch obvious breakage cheaply.

Examples:

- parser tests
- PAGE-XML read/write tests
- graph serialization tests
- feature-shape tests

These should be fast and narrow. They are useful, but they are not sufficient because they do not prove the user workflow still works.

### Level 1: Automatic End-To-End Regression

Purpose: verify that the full automatic path still works on a fixed evaluation set.

This is the current baseline. It runs:

- upload
- layout analysis
- PAGE-XML generation
- OCR
- evaluation against ground truth

This level should remain the required pre-commit quality gate for changes that can affect `app/`, `src/gnn_inference`, recognition, PAGE-XML generation, or evaluation code.

### Level 2: Scripted GUI Workflow Tests

Purpose: verify that the browser-facing workflow still works when driven like a real user.

These tests should eventually cover:

- manuscript upload through the UI
- page navigation
- save and recognize flows
- toggling recognition modes
- visibility of predicted overlays
- export flows

These tests are slower and more fragile than backend-only tests, so they should usually run in CI on pull request or nightly schedules rather than before every commit.

### Level 3: Human-In-The-Loop Evaluation

Purpose: measure how much manual intervention is still needed when a human uses the tool to correct model output.

This is essential for the repository vision. A layout model that is slightly less accurate but drastically easier to correct may be better for this product than one that scores better on a static benchmark but is painful to repair in the GUI.

### Level 4: Active-Learning Cycle Evaluation

Purpose: measure whether repeated correction plus fine-tuning actually reduces human effort over successive pages.

This is the evaluation level most closely tied to `VISION.md`.

The target experiment looks like this:

1. Run the current model on page 1.
2. Let a human correct the result.
3. Convert the corrections into fine-tuning data.
4. Fine-tune one or more models.
5. Run the updated model on page 2.
6. Measure whether correction effort decreases.
7. Repeat for the rest of the manuscript.

Success is not only lower CER or better segmentation metrics. Success is a downward trend in human effort while quality remains stable or improves.

## Metrics By Task

Future evaluation should be organized by the four tasks implied by the tool.

### 1. Character Detection Or Surrogate CRAFT Task

What we care about:

- missing character centers
- false character centers
- downstream effect on line segmentation quality
- amount of node correction needed by the human

Metrics to track:

- nodes added by human
- nodes deleted by human
- final node count
- precision and recall if node-level ground truth exists
- downstream line-level CER and segmentation metrics after correction

Reality check:

The repository currently logs node corrections in `app/app.py` to `node_corrections/`. This is useful but incomplete. It does not yet provide a full character-detection benchmark with node-level ground truth.

### 2. GNN Edge Classification For Text-Line Segmentation

What we care about:

- whether text lines are segmented correctly
- whether the graph is easy to correct when wrong

Metrics to track:

- line-level CER using PAGE-XML polygon matching
- PAGE-level CER
- line IoU thresholds
- edge additions by human
- edge deletions by human
- connected-component purity
- split and merge error counts

Reality check:

The current automatic evaluator already measures PAGE-level and line-level CER via `app/tests/evaluate.py`. Human edge corrections are not yet logged as first-class evaluation artifacts and should be added before claiming strong human-in-the-loop results.

### 3. Text-Region Grouping

What we care about:

- whether text lines are grouped into the correct text region
- whether region labeling can be corrected quickly

Metrics to track:

- region assignment accuracy when region labels exist
- region split count
- region merge count
- textbox label edits by human
- downstream export correctness in PAGE-XML

Reality check:

This task is supported in the UI, but there is no formal evaluation pipeline for it yet. Future work should add region-level ground truth and region-edit logging.

### 4. Text Recognition

What we care about:

- OCR quality on automatically segmented lines
- OCR quality after human corrections
- whether fine-tuning lowers correction burden

Metrics to track:

- character error rate
- optional word error rate if appropriate for the script and corpus
- per-line confidence summaries
- number of text edits made by the human
- time spent correcting text

Reality check:

The current automatic evaluator measures CER from recognized PAGE-XML output. It does not yet measure manual OCR correction effort.

## Human Effort Metrics

Because the vision is active learning, human effort is a first-class metric. Every future human-in-the-loop evaluation should try to capture:

- total session time
- time per page
- nodes added
- nodes deleted
- edges added
- edges deleted
- textbox label changes
- text edits
- number of save cycles
- whether recognition had to be re-run
- whether model fine-tuning was triggered after the page

These effort metrics should be recorded per page, not only per manuscript.

## Required Artifacts For Every Evaluation Run

Each serious evaluation run should produce a dedicated artifact folder. The exact root can change, but the contents should be consistent.

Recommended structure:

```text
app/tests/logs/<timestamp>_<run_name>/
  config.json
  summary.md
  metrics.json
  per_page.csv
  stdout.log
  predicted_page_xml/
  manual_events.jsonl
  fine_tune_metadata.json
```

Where each file means:

- `config.json`: dataset paths, model checkpoints, thresholds, resize settings, min-distance, commit SHA
- `summary.md`: short human-readable summary of the run
- `metrics.json`: machine-readable aggregate and per-task metrics
- `per_page.csv`: one row per page
- `stdout.log`: raw execution transcript
- `predicted_page_xml/`: frozen prediction outputs for auditability
- `manual_events.jsonl`: timestamped user actions for GUI or human-in-the-loop experiments
- `fine_tune_metadata.json`: what model was fine-tuned, on what data, for how long, using what checkpoint lineage

The current automatic test only writes `ci_eval_results_latest.txt` and `ci_eval_results_latest.json`. That is enough for the current regression gate, but future evaluations should graduate to the fuller structure above.

## Current Pre-Commit Gate

The repository currently uses a local git pre-commit hook to run the headless automatic evaluation:

```bash
cd app
conda activate gnn_layout
python -m unittest discover -s tests -p "test_ci_e2e.py" -v
```

The hook is versioned in `.githooks/pre-commit`. The local repository should be configured once with:

```bash
python scripts/install_git_hooks.py
```

The hook delegates to `scripts/run_precommit_eval.py`. That launcher first tries to find the `gnn_layout` interpreter directly, then falls back to `conda run -n gnn_layout python` if needed. If a contributor keeps the environment in a non-standard location, they should set `GNN_LAYOUT_PYTHON` to the full path of the desired interpreter.

The hook is meant to be the normal gate before commit, not a source of mystery failures. If someone intentionally needs to bypass it for one local commit, they can use `git commit --no-verify`. For one-off debugging of the hook wrapper itself, `SKIP_EVAL_HOOK=1` is also supported.

The gate currently checks these aggregate thresholds against the 15-page evaluation dataset:

- page CER <= 0.40
- line CER at IoU 0.50 <= 0.40
- line CER at IoU 0.75 <= 0.45
- line CER range <= 0.48
- worst single-page line CER at IoU 0.50 <= 0.55

These are regression thresholds, not research targets. They should only be tightened deliberately and with baseline evidence.

## Blueprint For Future GUI Evaluation

Future GUI evaluation should use a scripted browser driver and, when needed, a human operator protocol.

### Scripted GUI Evaluation Should Log

- page load time
- upload completion time
- save completion time
- recognition completion time
- any frontend error banners
- exported artifact paths

### Human-Guided GUI Evaluation Should Log

- operator identifier or anonymized operator code
- page start timestamp
- page end timestamp
- node edits
- edge edits
- region edits
- text edits
- model version shown to the operator
- whether a fine-tune happened before the page

### Files That Will Likely Need Instrumentation

- `app/frontend/src/components/ManuscriptViewer.vue`
- `app/app.py`

These are the files that currently mediate most user actions and save flows. If we want reliable human-effort evaluation, these files will need structured event logging rather than only ad hoc side effects.

## Blueprint For Active-Learning Evaluation

The core active-learning evaluation unit should be a manuscript ordered page sequence.

The protocol should look like this:

1. Choose a manuscript and freeze page order.
2. Run the current model on the first page.
3. Record automatic quality metrics.
4. Let the human correct the output in the GUI.
5. Record human-effort metrics.
6. Convert corrections into training data.
7. Fine-tune one or more target models.
8. Record training metadata.
9. Run the updated model on the next page.
10. Repeat until the manuscript ends.

The main expected curve is:

- human effort per page should trend downward
- output quality should remain stable or improve
- fine-tuning latency must be low enough to be practical between pages

That means future active-learning experiments must report at least:

- page index
- pre-correction quality
- post-correction quality
- correction effort
- fine-tune duration
- next-page quality after fine-tune

## Evaluation Templates

Every new evaluation effort should begin by filling out the following template.

### Evaluation Spec Template

- Name:
- Goal:
- Task scope:
- Dataset:
- Human involved:
- Fine-tuning involved:
- Primary metrics:
- Secondary metrics:
- Acceptance rule:
- Artifacts written:
- Owner files:

### Per-Run Checklist

- Record commit SHA.
- Record model checkpoint paths.
- Record dataset paths.
- Record resize and preprocessing settings.
- Record whether GPU or CPU was used.
- Record the exact command used.
- Save outputs and logs in one folder.
- Save aggregate and per-page metrics.

## What We Should Not Do

To keep evaluation honest, avoid the following:

- do not claim active-learning improvement from one isolated before-vs-after comparison
- do not report only automatic quality without correction effort for human-in-the-loop workflows
- do not overwrite prediction outputs without saving a run artifact
- do not tighten regression thresholds without recording the baseline that justified it
- do not mix datasets, settings, or model checkpoints without saving metadata

## Near-Term Recommended Additions

The following are the next high-value evaluation improvements, in order:

1. Add structured logging for edge edits, textbox edits, and OCR text edits.
2. Save timestamped run folders instead of only `latest` reports.
3. Add a scripted GUI smoke test for upload, save, and page navigation.
4. Add a formal region-grouping evaluation dataset and metrics.
5. Add a manuscript-sequential active-learning benchmark protocol.

## Definition Of Success

For this repository, evaluation maturity means:

- every code change can be screened by an automatic end-to-end gate
- every research claim can be tied to saved artifacts and exact settings
- human-in-the-loop experiments can measure effort, not just accuracy
- active-learning experiments can prove that manual effort drops from early pages to later pages after fine-tuning
- time taken for the entire pipeline to run end-to-end (eg: optimize the GNN pipeline for speed)

That is the standard future evaluation should be built to.
