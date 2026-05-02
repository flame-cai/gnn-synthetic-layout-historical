# Deduplicate GNN Checkpoint Loading

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with [PLANS.md](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/PLANS.md).

## Purpose / Big Picture

After this change, the repository will load GNN checkpoints for both automatic inference and the semi-automatic app without needing copied `gnn_training` or `gnn_data_preparation` packages under multiple directories. The user-visible proof is simple: loading the existing pretrained checkpoint from `app/` and from `src/gnn_inference/` should still work, but both paths should resolve model classes and preprocessing code from the single source of truth in `src/gnn_training/`.

## Progress

- [x] (2026-04-15 17:05 IST) Confirmed the duplication root cause: checkpoints store full model objects, so `torch.load` requires the original `gnn_training.training.models.gnn_models` import path.
- [x] (2026-04-15 17:08 IST) Confirmed that `app/gnn_training` and `src/gnn_inference/gnn_training` only exist to satisfy that deserialization path, while inference code also duplicates `gnn_data_preparation`.
- [x] (2026-04-15 17:18 IST) Added deterministic `src/` path bootstrapping in both inference entrypoints before deserialization and shared imports.
- [x] (2026-04-15 17:19 IST) Repointed both inference implementations at `src/gnn_training/gnn_data_preparation`.
- [x] (2026-04-15 17:20 IST) Removed duplicated `gnn_training` and `gnn_data_preparation` source files from `app/` and `src/gnn_inference/`.
- [x] (2026-04-15 17:22 IST) Ran static verification: changed modules compile, shared imports resolve textually, and duplicate source trees no longer contain Python files.
- [x] Run runtime smoke checks proving checkpoint loading still works from both execution roots in an environment with `torch` installed.

## Surprises & Discoveries

- Observation: legacy checkpoints currently save the full `model` object rather than a `state_dict`.
  Evidence: [src/gnn_training/training/utils.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/src/gnn_training/training/utils.py:31) writes `'model': model`, and both inference paths load `checkpoint['model']`.

- Observation: this can be fixed without changing the shipped checkpoints, as long as `src/` is placed ahead of local script directories on `sys.path`.
  Evidence: the deserialization import path is the top-level package name `gnn_training`, so the first matching `gnn_training` package on `sys.path` wins.

## Decision Log

- Decision: scope this refactor to deduplicating the checkpoint-loading and inference-preprocessing packages first, rather than redesigning checkpoint format in the same change.
  Rationale: it removes the duplicated packages immediately while keeping existing pretrained checkpoints usable.
  Date/Author: 2026-04-15 / Codex

- Decision: keep `src/gnn_training` as the single source of truth.
  Rationale: training already lives there, and legacy checkpoints were trained against that package path.
  Date/Author: 2026-04-15 / Codex

## Outcomes & Retrospective

The repository now has a single Python source of truth for GNN model definitions and inference-time preprocessing: `src/gnn_training/`. Both inference entrypoints now explicitly insert `src/` at the front of `sys.path`, which is the key compatibility step for legacy checkpoints that still deserialize a full `gnn_training.training.models.gnn_models` object. The copied Python packages under `app/` and `src/gnn_inference/` were removed.

The remaining gap is runtime validation. The shell available during implementation did not have PyTorch installed, so the actual `torch.load` smoke test could not be executed here. That should be the next check in a configured environment before merging.

## Context and Orientation

The current checkpoint loading path is split across three places. Training code in `src/gnn_training/training/` defines model classes and saves checkpoints. Automatic inference in `src/gnn_inference/` loads those checkpoints. The Flask app in `app/` also loads those checkpoints for page-level prediction. Because training checkpoints currently serialize a whole Python object, Python must be able to import the exact module path that defined the class when the checkpoint was written. That is why the repository currently contains copied `gnn_training` packages under `app/` and `src/gnn_inference/`.

The duplicated preprocessing packages (`app/gnn_data_preparation` and `src/gnn_inference/gnn_data_preparation`) exist for the same practical reason: inference entrypoints import them locally instead of using the canonical package under `src/gnn_training/gnn_data_preparation`.

## Plan of Work

Update `app/gnn_inference.py` so it computes the repository `src/` path from `__file__`, inserts that directory at the front of `sys.path`, and imports preprocessing utilities from `gnn_training.gnn_data_preparation`. Make the same change in `src/gnn_inference/gnn_inference.py`, but compute `src/` as the parent directory of `src/gnn_inference/`.

Once both entrypoints import from `src/gnn_training`, delete the copied Python sources under `app/gnn_training`, `app/gnn_data_preparation`, `src/gnn_inference/gnn_training`, and `src/gnn_inference/gnn_data_preparation`. Keep unrelated files untouched.

Then run targeted smoke checks that import the inference modules from their native working directories and call the lazy model-loading path against the shipped pretrained checkpoint.

## Concrete Steps

From the repository root:

    python -m py_compile app\gnn_inference.py src\gnn_inference\gnn_inference.py

Expected result: command exits with code 0 and no output.

From the repository root:

    rg -n "from gnn_training\.gnn_data_preparation|sys\.path\.insert\(0, str\(SRC_ROOT\)\)" app\gnn_inference.py src\gnn_inference\gnn_inference.py

Expected result: both files show `sys.path.insert(0, str(SRC_ROOT))` and `from gnn_training.gnn_data_preparation ...` imports.

Runtime smoke checks still to run in a configured environment:

From working directory `app`:

    python -c "import gnn_inference; gnn_inference.load_model_once('./pretrained_gnn/v2.pt', './pretrained_gnn/gnn_preprocessing_v2.yaml'); print('ok')"

From working directory `src\gnn_inference`:

    python -c "import gnn_inference; print('import ok')"

## Validation and Acceptance

Acceptance is:

1. From working directory `app`, importing `gnn_inference` and calling `load_model_once` with `app/pretrained_gnn/v2.pt` succeeds without `app/gnn_training` or `app/gnn_data_preparation` source files.
2. From working directory `src/gnn_inference`, importing `gnn_inference` and loading `pretrained_gnn/v2.pt` succeeds without `src/gnn_inference/gnn_training` or `src/gnn_inference/gnn_data_preparation` source files.
3. Both inference modules import preprocessing code from `src/gnn_training/gnn_data_preparation`.
4. `python -m py_compile app\gnn_inference.py src\gnn_inference\gnn_inference.py` succeeds.

## Idempotence and Recovery

The refactor is additive until file deletion. If a smoke check fails, restore the deleted duplicate package files from git and inspect which import still resolves locally instead of through `src/`.

## Artifacts and Notes

The key artifact is the import-path change itself: inference should resolve `gnn_training` from `src/`, not from copied local packages.

Implementation note added on 2026-04-15: runtime `torch.load` verification could not be completed in-session because the available shell Python lacked `torch`. Static verification was completed instead.

## Interfaces and Dependencies

The following interfaces must remain available after the change:

- `gnn_training.training.models.gnn_models` for legacy checkpoint deserialization.
- `gnn_training.gnn_data_preparation.config_models.DatasetCreationConfig`
- `gnn_training.gnn_data_preparation.graph_constructor.create_input_graph_edges`
- `gnn_training.gnn_data_preparation.feature_engineering.get_node_features`
- `gnn_training.gnn_data_preparation.feature_engineering.get_edge_features`
- `gnn_training.gnn_data_preparation.utils.setup_logging`
