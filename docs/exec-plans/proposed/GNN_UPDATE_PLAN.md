# Safe GNN Checkpoint Loading And Migration Plan

## Summary
- Keep using the existing `gnn_layout` conda environment and the current older PyTorch stack for this checkpoint hardening pass.
- Do not use `gnn_layout_new`, do not upgrade PyTorch in this plan, and do not introduce `safetensors`.
- Move GNN checkpoints from full Python-object pickle checkpoints to tensor-only `state_dict` checkpoints.
- Keep existing production paths unchanged: `app/pretrained_gnn/v2.pt`, `src/gnn_inference/pretrained_gnn/v2.pt`, and their paired YAML files.
- Migrate the current trusted `v2.pt` once, without retraining.
- Run the full app pre-commit pipeline against the migrated production `v2.pt`.
- Separately generate 5000 toy synthetic samples, process them, train a toy MPNN, and smoke-test safe load/inference on that toy checkpoint.

## Key Changes
- Add shared GNN checkpoint helpers under `src/gnn_training/training/`:
  - Build GNN models from explicit `model_name`, `node_feature_dim`, `edge_feature_dim`, and checkpoint-stored model config.
  - Save checkpoints with:
    - `checkpoint_format_version: 2`
    - `checkpoint_kind: "gnn_edge_classifier_state_dict"`
    - `model_name`
    - `model_config`
    - `node_feature_dim`
    - `edge_feature_dim`
    - optional `preprocessing_config_sha256`
    - optional canonical `preprocessing_config`
    - `model_state_dict`
    - optional `optimizer_state_dict`
    - `epoch`, `metrics`, and version/logging metadata.
  - Load checkpoints using `torch.load(..., weights_only=True)` only.

- Update GNN training:
  - Replace `'model': model` checkpoint saves with `model_state_dict`.
  - Update final evaluation reload to reconstruct the model and call `load_state_dict`.
  - Add an optional `--preprocessing_config` CLI argument so training checkpoints can record the preprocessing YAML hash.
  - Keep optimizer state for training resume and future iterative GNN fine-tuning; inference ignores it.
  - Add logging for checkpoint path, format version, model name, feature dims, epoch, metrics, and preprocessing hash presence.

- Update app and src inference:
  - Replace `checkpoint["model"]` loading with the shared safe loader.
  - Keep existing constants and filenames unchanged.
  - Validate external preprocessing YAML against the checkpoint hash when present.
  - Validate runtime feature dimensions before model inference; fail early with a clear error if YAML/features do not match the weights.
  - Log one concise model-load event per process/cache load, not per page.

- Migration:
  - Add `scripts/migrate_gnn_checkpoint.py`.
  - Require an explicit trusted-legacy flag before using `weights_only=False`.
  - Load the current trusted `v2.pt` once, extract `model.state_dict()`, preserve `optimizer_state_dict` if present, attach `SplineCNN` metadata and the current preprocessing YAML hash, save a safe v2 checkpoint, and copy the same migrated bytes to both pretrained GNN folders.
  - Verify the migrated checkpoint loads with `weights_only=True`.
  - Verify old-model vs reconstructed-model logits match on a deterministic synthetic graph before replacing artifacts.

- Dependency compatibility:
  - Do not change `requirements.txt`, `requirements_cpu.txt`, or the active conda environment as part of this plan.
  - Keep running verification in `gnn_layout`.
  - Treat `torch.load(..., weights_only=True)` on the current older PyTorch as a practical hardening step, not as the full upstream CVE remediation.
  - Document that GitHub may continue flagging the pinned PyTorch dependency until a separate compatible PyTorch upgrade is possible.
  - Treat PyG processed dataset cache loading separately as trusted local training data, because it is not a shipped model checkpoint.

## Verification
- Unit tests:
  - Round-trip save/load a tiny MPNN checkpoint using `weights_only=True`.
  - Assert new checkpoints do not contain a `model` object key.
  - Assert legacy/full-object checkpoints produce a clear migration error in runtime loaders.
  - Assert preprocessing hash mismatch raises a clear error.
  - Assert feature-dimension mismatch raises before inference.

- Migration validation:
  - Run the migration script on current `app/pretrained_gnn/v2.pt`.
  - Confirm both migrated `v2.pt` copies have identical SHA256 hashes.
  - Confirm both paired YAML files remain semantically unchanged and identical.
  - Smoke-load from `app` and from `src/gnn_inference`.

- Toy MPNN verification:
  - Add `scripts/run_gnn_safe_checkpoint_smoke.py`.
  - Under a temp directory, generate 5000 synthetic samples from a temporary copy of `src/configs/synthetic.yaml`.
  - Split generated raw files into temp train and val/test directories.
  - Process them with `src/gnn_training/gnn_data_preparation/main_create_dataset.py`.
  - Train `models_to_run: ["MPNN"]` for a short smoke run, with no visualization and no performance target.
  - Load the trained MPNN checkpoint with `weights_only=True`.
  - Run one direct tensor inference pass on a processed graph and one src inference-path load/predict smoke check.

- Full app gate:
  - Run only against the migrated production `v2.pt`, not the toy MPNN:
    ```powershell
    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v
    ```
  - This verifies the real app pipeline still loads and runs with the migrated safe checkpoint.

## Assumptions
- `gnn_layout` remains the active environment for this work.
- No PyTorch upgrade, `gnn_layout_new` usage, or `safetensors` adoption is part of this plan.
- This plan hardens GNN checkpoint loading/saving behavior but may not silence GitHub's PyTorch dependency alert while the older PyTorch pin remains.
- The toy MPNN is a load/save/inference smoke test only; it is not expected to pass production quality thresholds.
- The preprocessing YAML remains tightly coupled to weights, but the coupling becomes explicit through checkpoint hash validation.
- No CRAFT, OCR, text-line strategy, or app workflow behavior should change beyond GNN checkpoint loading.
