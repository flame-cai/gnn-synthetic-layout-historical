# Runbook

One iteration should test one hypothesis.

1. Read `IDEAS.md` and pick one idea id.
2. Change only the strategy, unwrapping, or selector code needed for that idea.
3. Run unit tests:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_circular_ocr_unit -v
```

4. Run the experiment gate:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_circular_ocr_precommit_e2e -v
```

5. For pre-commit profile coverage, run:

```powershell
$env:PRECOMMIT_EVAL_PROFILE='circular_layout'
conda run -n gnn_layout python scripts/run_precommit_eval.py
```

6. Inspect latest artifacts under `app/tests/logs/`.
7. Report metric deltas and whether the idea should be accepted, revised, deferred, or rejected.

The user remains commit owner during the supervised phase.

