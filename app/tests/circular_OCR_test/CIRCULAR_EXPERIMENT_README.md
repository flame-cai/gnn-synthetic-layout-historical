# Circular OCR Experiment

This directory contains the opt-in harness for curved, circular, and vertical text-line preparation before OCR. It does not change GUI save or recognition behavior by default.

Run the circular gate:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_circular_ocr_precommit_e2e -v
```

Artifacts are written under `app/tests/logs/<timestamp>_circular_ocr_eval_dataset_v2/`. Latest pointers are:

- `app/tests/logs/circular_ocr_latest.md`
- `app/tests/logs/circular_ocr_latest.json`
- `app/tests/logs/circular_ocr_latest.txt`

Durable checked-in state lives here: `IDEAS.md`, `BASELINE.md`, `baseline.json`, `RUNBOOK.md`, `STRATEGY_CONTRACT.md`, and `FAILURE_MODES.md`.

