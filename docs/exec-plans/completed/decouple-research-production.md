# Decouple Research Promotion From Production Strategy Adoption

## Summary

Separate text-line segmentation strategy control into two explicit lifecycles:

- `research harness lifecycle`: compare `benchmark_strategy_name` vs `proposed_strategy_name`, promote within the verifier harness, preserve research history.
- `production app lifecycle`: independently choose one `production_strategy_name` for the GUI/runtime.

This removes the current coupling where a research promotion silently changes the live GUI default. It also creates a clear documented workflow for two different actions:

1. `promote in harness`
2. `adopt in production`

Initial behavior after the refactor must remain unchanged:

- research benchmark stays `legacy_axis_bound_v1`
- research proposed stays `local_tangent_band_v1`
- production app stays pinned to `legacy_axis_bound_v1`

Existing manuscripts/pages are not migrated. Future saves/regenerations use the then-current `production_strategy_name`.

## Key Changes

### 1. Split strategy config into research and production roles

Refactor `app/recognition/line_segmentation/strategy_config.py` so it becomes the single source of truth for both domains.

New checked-in payload shape:

- `benchmark_strategy_name`
- `proposed_strategy_name`
- `production_strategy_name`
- `research_promotion_history`
- `production_adoption_history`

Getter/setter contract:

- keep `get_benchmark_strategy_name()`
- keep `get_proposed_strategy_name()`
- add `get_production_strategy_name()`
- keep normalized render/write helpers, extended to the new payload

Default checked-in values:

- `benchmark_strategy_name = legacy_axis_bound_v1`
- `proposed_strategy_name = local_tangent_band_v1`
- `production_strategy_name = legacy_axis_bound_v1`
- empty research/adoption history lists

Design rule:

- `benchmark_strategy_name` means “current verifier research baseline”
- `production_strategy_name` means “current live app default”

No code outside the verifier harness may use `benchmark_strategy_name` as the live app selector after this refactor.

### 2. Make research promotion harness-only

Keep `scripts/promote_text_line_strategy.py`, but narrow its meaning and write behavior.

New behavior:

- validate research evidence exactly as today
- on apply, update only:
  - `benchmark_strategy_name`
  - `proposed_strategy_name`
  - `research_promotion_history`
- never modify:
  - `production_strategy_name`
  - `production_adoption_history`

Messaging changes:

- all script output must say this promotes a strategy within the research harness
- any text that implies it affects the GUI/app default must be removed

History entry behavior:

- preserve current evidence linkage pattern for research promotions
- continue recording artifact paths, metrics, timestamps, and tool identity

### 3. Add a separate production-adoption workflow

Add a new explicit script for production adoption, for example `scripts/adopt_text_line_strategy_for_app.py`.

Script contract:

- input: `--strategy <registered_strategy_name>`
- optional: `--reason <string>`
- supports dry run and `--apply`
- validates the named strategy is registered
- writes only:
  - `production_strategy_name`
  - `production_adoption_history`
- does not require verifier evidence
- does not mutate benchmark/proposed research roles
- is idempotent if the app is already pinned to that strategy

Production adoption history entries should record:

- adopted strategy name
- previous production strategy name
- adoption timestamp
- author/tool
- optional human reason

This creates a safe two-step operational model:

- research can iterate independently
- production changes happen only by explicit separate adoption

### 4. Switch app runtime defaults to production strategy only

Update all app/runtime code paths that currently infer a live strategy from the research benchmark.

Required changes:

- in `app/gnn_inference.py`, replace `get_benchmark_strategy_name()` with `get_production_strategy_name()` for the app default
- search for any other non-test app/runtime imports of the benchmark getter and move them to the production getter if they control live app behavior
- keep verifier/test harness code on benchmark/proposed getters

Important non-change for safety:

- original scope: do not change GUI OCR crop behavior in this refactor
- 2026-05-15 follow-up: production OCR callers now route through `app/recognition/line_segmentation/ocr_crops.py`; with the current `legacy_axis_bound_v1` production pin this still produces the masked PAGE `Coords` crop
- GUI OCR inference and active-learning training still read saved PAGE `Coords`; sibling metadata only decides whether the derived OCR crop remains masked or uses a local-tangent unwrap after explicit production adoption
- no existing manuscripts are migrated by the crop-layer refactor

This keeps the change small, bounded, and non-disruptive.

### 5. Preserve existing manuscript/page behavior

Adopt a strict no-migration policy.

Rules:

- do not rewrite existing PAGE XML
- do not rebuild existing OCR line images
- do not invalidate active-learning lineage
- do not auto-regenerate historical manuscripts when production strategy changes

Operational result:

- strategy adoption affects only future layout saves/regenerations
- already saved pages continue behaving as they do now

### 6. Update docs so users and future agents can operate both workflows safely

This refactor must include doc updates as a first-class deliverable, not a follow-up.

Required documentation updates:

- `EVAL.md`
  - explain that verifier promotion is research-only
  - document the distinction between harness promotion and production adoption
  - update workflow sections so users know which command does which
- `README.md`
  - explain which strategy the app uses in production
  - explain that research harness roles are separate from the production app role
  - add a concise operator-facing section for “research promotion vs production adoption”
- `AGENTS.md`
  - update source-of-truth guidance so future agents know:
    - where research role state lives
    - where production role state lives
    - which script promotes in harness
    - which script adopts to production
  - explicitly warn that harness promotion must not be treated as app rollout
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-workflow.md`
  - split the workflow into:
    - harness promotion
    - production adoption
  - include prerequisites, dry run/apply commands, expected config changes, and what does not change
- `docs/pipeline-improvement/text-line-segmentation/local-tangent-band-v1-architecture.md`
  - update “Current Gate Behavior” / rollout wording so it no longer implies research benchmark equals app default
- any production-facing planning or evaluation docs that currently describe promotion as if it changes the app
  - update terminology to use:
    - `promotion` for research harness
    - `adoption` for production app

Doc content requirements:

- include exact command examples for both workflows
- state that production adoption is explicit and separate
- state that existing pages are not migrated automatically
- state that current GUI OCR crop behavior is unchanged by this refactor

## Public Interfaces / Scripts / Types

Add or change these interfaces:

- `get_production_strategy_name()` in line segmentation config
- strategy config payload gains:
  - `production_strategy_name: str`
  - `research_promotion_history: list`
  - `production_adoption_history: list`
- `scripts/promote_text_line_strategy.py`
  - semantics: research harness promotion only
- new script:
  - `scripts/adopt_text_line_strategy_for_app.py --strategy <name> [--reason ...] [--apply]`

Config rendering/writing must remain stable and readable in a fresh clone.

## Test Plan

### Config tests

- verify normalization, rendering, and writing of the expanded config payload
- verify default values:
  - benchmark = `legacy_axis_bound_v1`
  - proposed = `local_tangent_band_v1`
  - production = `legacy_axis_bound_v1`

### Research promotion tests

- dry run changes only research fields
- apply changes only:
  - benchmark
  - proposed
  - research history
- production strategy remains unchanged
- stale/missing/mismatched evidence checks still work
- idempotent re-run still works

### Production adoption tests

- dry run changes only production fields
- apply changes only:
  - production strategy
  - production adoption history
- benchmark/proposed remain unchanged
- unknown strategy is rejected
- idempotent adoption works

### App integration tests

- assert the app live default uses `get_production_strategy_name()`
- assert the app no longer imports the research benchmark as its live selector
- simulate:
  - research benchmark = `local_tangent_band_v1`
  - production = `legacy_axis_bound_v1`
  - and verify app still resolves to legacy
- simulate:
  - research benchmark = `legacy_axis_bound_v1`
  - production = `local_tangent_band_v1`
  - and verify app resolves to local tangent

### Documentation regression checks

- unit tests that currently assert benchmark-linked app behavior must be updated to assert production-linked app behavior
- if there are tests that inspect command/help text or workflow docs, update them to reflect the two-step lifecycle
- add at least one unit test that asserts the promotion script does not touch production strategy

## Assumptions and Defaults

- Shared implementation code remains shared between research and production.
- Only rollout control is separated in this plan.
- GUI OCR crop-mode integration is intentionally deferred; this plan is about safe lifecycle separation.
- Existing manuscripts/pages are not migrated.
- Production remains pinned to `legacy_axis_bound_v1` until someone explicitly adopts another strategy.
- Research may continue promoting new harness benchmarks without any direct GUI effect.

## Implementation Order

1. Extend strategy config schema and getters to represent research and production separately.
2. Refactor the existing promotion script so it updates research roles only.
3. Add the explicit production-adoption script and production adoption history.
4. Switch app/runtime live defaults to `get_production_strategy_name()`.
5. Update and expand tests to enforce the separation boundary.
6. Update all relevant docs so users and future agents can safely run both workflows.

## Acceptance Criteria

- Research promotion can occur without changing GUI production behavior.
- Production adoption can occur without changing research benchmark/proposed roles.
- The app behaves exactly as it does today immediately after the refactor.
- Operators have two explicit documented commands/workflows:
  - one for harness promotion
  - one for production adoption
- Future agents reading the repository docs can correctly determine:
  - how to promote in harness
  - how to adopt to production
  - what files/scripts are the source of truth
  - what behavior remains unchanged
