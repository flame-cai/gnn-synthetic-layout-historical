# Future DeepSeek and Sarvam Integration Guide

## Purpose

This document is for future agents adding DeepSeek or Sarvam to the downstream
OCR experiment.

The experiment compares complete OCR methods on historical manuscript pages.
For every evaluated page, a method may predict:

- text-line geometry;
- Unicode transcription;
- a valid PAGE-XML representation;
- page-level OCR and layout metrics.

The experiment currently evaluates manuscripts with the directory and labeling
contract used by:

```text
app/input_manuscripts/yajn
app/input_manuscripts/dense
app/input_manuscripts/circle_new
```

Future manuscripts with the same contract must work without provider-specific
or manuscript-specific branches.

Production code under `app/` is read-only for this experiment. Provider
integration belongs entirely under:

```text
experiments/downstream_ocr/
```

Do not change production application behavior to add an experimental provider.

## Current Experiment Architecture

The paid-inference workflow has two deliberately separate phases.

### Phase 1: acquire every page once

The acquisition command is:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli prepredict-vlms `
  --manuscript-root app\input_manuscripts\yajn `
  --manuscript-root app\input_manuscripts\dense `
  --manuscript-root app\input_manuscripts\circle_new `
  --output-root <immutable-cache-root> `
  --provider-id <provider>
```

Acquisition runs over the complete discovered page set. It is independent of
fold creation and fold selection.

### Phase 2: evaluate cached predictions

`run-methods` must not call a paid API. It receives:

```powershell
--vlm-predictions-root <immutable-cache-root>
```

It validates the provider cache before doing local OCR work, materializes only
the relevant test pages for each fold, and evaluates those PAGE-XML files.

All compared methods in a fold use the exact same test-page IDs. Adding a
provider must not add a provider-specific split or silently omit failed pages.
A provider failure is represented by an empty PAGE prediction with a retained
failure status.

## Source-of-Truth Modules

Future agents should first read these files completely:

```text
experiments/downstream_ocr/adapter.py
experiments/downstream_ocr/vlm_providers.py
experiments/downstream_ocr/vlm_cache.py
experiments/downstream_ocr/runners.py
experiments/downstream_ocr/reporting.py
experiments/downstream_ocr/tests/test_vlm_cache.py
```

Their responsibilities are:

| Module | Responsibility |
|---|---|
| `adapter.py` | Shared Gemini-style prompt and normalized JSON-to-PAGE conversion |
| `vlm_providers.py` | Provider registry, credentials metadata, SDK calls, normalized usage response |
| `vlm_cache.py` | Pay-once page acquisition, retries, fingerprints, immutable terminal results |
| `runners.py` | Offline fold materialization and evaluation dispatch |
| `reporting.py` | Method registry presentation and unique acquisition accounting |
| `tests/test_vlm_cache.py` | Cache, prompt, failure, and offline-run invariants |

Provider transport details must not leak into `runners.py`. The runner should
know only that a registered method has a validated cache containing PAGE-XML
and statuses.

## Non-Negotiable Experiment Invariants

### 1. Never acquire by fold

One provider prediction is acquired per:

```text
(provider, exact model, manuscript, page, request contract)
```

Folds only select from those predictions. Never place provider calls inside a
fold loop.

### 2. Do not silently repay

The cache treats successes and exhausted failures as terminal. Re-running the
acquisition command validates and reuses them.

If a synchronous request was interrupted after its page directory was created,
the paid state is ambiguous. Do not automatically submit a replacement
request.

For an asynchronous provider, persist the remote job identifier immediately.
Resume polling or downloading the existing job. Do not create another remote
job merely because the local process restarted.

### 3. Fingerprint every result-defining input

The request fingerprint must cover at least:

- cache schema version;
- manuscript ID and page ID;
- provider ID and method ID;
- exact model ID;
- exact input-contract version;
- exact output-adapter version;
- prompt hash, when a prompt is part of the method;
- page-image hash;
- template PAGE-XML hash;
- provider request settings that can affect predictions;
- language and output-format settings;
- reasoning or thinking configuration, if applicable.

Changing any of these requires a new acquisition namespace or cache root.
Never reinterpret an old response under a new adapter without recording a new
derived-artifact version.

### 4. Preserve the raw response

Retain enough provider-native evidence to audit the conversion:

- request metadata without secrets;
- every attempt or asynchronous state transition;
- provider response or downloaded result;
- normalized intermediate JSON;
- generated PAGE-XML;
- provider/model identifiers;
- response or job identifier;
- token or page billing metadata when available;
- terminal status and error;
- elapsed time.

API keys, authorization headers, and signed download URL query strings must
never be written to the cache.

### 5. Keep failures in the denominator

Authentication errors, timeouts, invalid JSON, missing geometry, rejected
files, and provider-side failures must not cause the page to disappear from
evaluation.

After retry exhaustion, write an empty PAGE prediction and retain the precise
status. Valid-output rate and OCR/layout metrics must reflect the failure.

### 6. Disable SDK retries

The experiment owns retry accounting. Provider SDK automatic retries must be
disabled, otherwise the recorded attempt count and actual paid request count
can diverge.

The current CLI definition of three retries means:

```text
one initial attempt + at most three retries = at most four attempts
```

### 7. Keep secrets and network access out of `run-methods`

Only `prepredict-vlms` may load provider credentials. `run-methods` must remain
fully offline with respect to paid providers.

### 8. Count acquisition usage once

A page may appear in multiple folds. API usage and cost must be read from the
unique acquisition cache, not summed from fold-materialized copies.

Metrics may include repeated fold occurrences according to the established
evaluation protocol. Provider billing must not.

## Recommended Modularization Before Adding Sarvam

The current provider interface is optimized for synchronous, prompt-compatible
VLM calls:

```python
invoke_provider(...) -> VlmProviderResponse
```

That is appropriate for providers that accept one page image and the shared
prompt and synchronously return the expected JSON.

Sarvam Document Digitization is asynchronous and returns provider-native
document artifacts. Do not force that workflow into a fake synchronous chat
call. Before adding Sarvam, separate three concepts.

### Provider specification

A provider specification should describe immutable method identity, not perform
I/O. A future shape could include:

```python
@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    method_id: str
    display_name: str
    model_id: str
    api_key_env: str
    acquisition_kind: Literal["sync_prompt_vlm", "async_document_ocr"]
    input_contract: str
    input_contract_version: int
    output_adapter_id: str
    output_adapter_version: int
    request_settings: Mapping[str, JSONValue]
```

The exact type names may differ. The important point is that method-defining
settings are explicit, immutable, serializable, and fingerprinted.

### Acquisition driver

Transport should be behind an experiment-owned protocol:

```python
class AcquisitionDriver(Protocol):
    def preflight(self, request: AcquisitionRequest) -> None: ...

    def acquire(
        self,
        request: AcquisitionRequest,
        *,
        checkpoint: Callable[[RemoteState], None],
    ) -> ProviderArtifact: ...
```

The cache remains responsible for:

- page discovery;
- immutable local directories;
- fingerprints;
- retry policy;
- atomic checkpoint writes;
- terminal results;
- failure PAGE-XML;
- manifests.

The driver remains responsible for:

- provider SDK or HTTP details;
- one logical provider operation;
- provider response IDs;
- remote asynchronous state;
- usage metadata;
- raw provider artifacts.

The `checkpoint` callback lets an asynchronous driver atomically persist a job
ID and state transitions without giving the driver ownership of cache layout.

Synchronous Gemini/OpenAI/Claude drivers can implement this protocol with one
request and no intermediate remote state.

### Output adapter

Provider transport and PAGE conversion are different responsibilities:

```python
class ProviderOutputAdapter(Protocol):
    def to_page(
        self,
        artifact: ProviderArtifact,
        *,
        template_page: PageXmlPage,
    ) -> NormalizedProviderPage: ...
```

Use at least two adapter families:

- `shared_prompt_json`: parses the existing Gemini-style `regions/lines`
  payload and delegates to the shared JSON-to-PAGE converter.
- `sarvam_document_json`: parses a pinned Sarvam page-output schema and maps
  documented line geometry and transcription into `PageXmlPage`.

The cache orchestration should select the adapter from the provider
specification. It should not contain `if provider == "sarvam"` parsing logic.

## DeepSeek Integration

### Current eligibility

As of July 2026, DeepSeek V4-Flash must not be added to this image OCR
benchmark.

Official DeepSeek documentation identifies `deepseek-v4-flash` and
`deepseek-v4-pro`, but describes V4 as text-only. Its Anthropic-compatible API
marks image message content as unsupported. DeepSeek's documented Copilot
vision behavior sends an image to another installed vision model and forwards
that model's textual description to DeepSeek.

Official references:

- [DeepSeek API models and pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [DeepSeek Anthropic API compatibility](https://api-docs.deepseek.com/guides/anthropic_api)
- [DeepSeek V4 Copilot vision proxy](https://api-docs.deepseek.com/quick_start/agent_integrations/github_copilot)

A Claude, GPT, Gemini, or other vision proxy would make the effective OCR
method:

```text
vision proxy -> textual description -> DeepSeek
```

That is not `DeepSeek V4-Flash end-to-end`. Do not register or report it under
that name.

### Eligibility gate for a future DeepSeek vision release

DeepSeek becomes eligible only when official documentation and the official
API establish all of the following:

1. The exact DeepSeek model accepts image input directly.
2. Image content is processed by that DeepSeek model, not an undocumented or
   configurable proxy model.
3. The API returns text or structured JSON suitable for the common output
   adapter.
4. The model ID and multimodal request format can be pinned.
5. Usage metadata and retry behavior can be audited.

Record the documentation URL and verification date in the integration change.
Use a fake image request in a non-production test account before registering
the provider, but never put a real paid smoke call in the unit-test suite.

### Implementation steps after DeepSeek becomes vision-capable

1. Add a registry entry such as:

   ```text
   provider_id: deepseek
   method_id: deepseek_e2e
   model_id: exact documented vision model ID
   api_key_env: DEEPSEEK_API_KEY
   acquisition_kind: sync_prompt_vlm
   input_contract: page_image_then_exact_shared_prompt
   output_adapter_id: shared_prompt_json
   ```

2. Add a DeepSeek driver in the provider transport module.

3. If the official endpoint remains OpenAI-compatible, reuse the installed
   OpenAI SDK with the official DeepSeek base URL. Do not reuse the OpenAI
   provider function directly; create a named DeepSeek driver so endpoint,
   model, usage mapping, and request settings remain explicit.

4. Disable SDK retries.

5. Send the unchanged page image and the exact
   `VLM_END_TO_END_PROMPT`. Preserve image-first, prompt-second ordering when
   the API supports ordered multimodal content.

6. Pin JSON-output, temperature, reasoning/thinking, maximum-output, and other
   behavior-affecting settings in the provider specification and request
   fingerprint.

7. Normalize provider usage fields into:

   ```text
   input_tokens
   output_tokens
   total_tokens
   response_id
   finish_reason
   ```

8. Use the existing shared prompt JSON output adapter. Do not create a
   DeepSeek-specific PAGE writer if the response contract is identical.

9. Add `DEEPSEEK_INPUT_USD_PER_1M_TOKENS` and
   `DEEPSEEK_OUTPUT_USD_PER_1M_TOKENS` reporting support through the existing
   provider-neutral pricing convention.

10. Update CLI choices, report tables, documentation, and tests through the
    provider registry rather than hard-coded method lists.

### DeepSeek tests

At minimum, add tests proving:

- the exact model ID is pinned;
- an explicit capability guard rejects text-only models;
- the exact shared prompt is passed unchanged;
- the original image bytes are supplied directly;
- no proxy-provider identifier appears in the request;
- automatic SDK retries are disabled;
- response and usage fields normalize correctly;
- malformed or non-JSON output follows the common retry/failure path;
- rerunning a terminal page does not invoke the driver;
- `run-methods` uses the cache without importing or calling the driver.

## Sarvam Integration

### Sarvam is a different method contract

Sarvam Vision is a document-intelligence model intended for Indic OCR,
including historical documents and Sanskrit. Its official Document
Digitization workflow accepts PDF, PNG, JPG, or ZIP inputs, supports Sanskrit
through language code `sa-IN`, and always includes structured page-level JSON
with its HTML or Markdown output.

The official API is asynchronous:

```text
create job
obtain upload URL / upload
start job
poll or await completion
obtain download URL
download output archive
parse page JSON
```

Official references:

- [Sarvam Vision model](https://docs.sarvam.ai/api-reference-docs/getting-started/models/sarvam-vision)
- [Document Digitization overview](https://docs.sarvam.ai/api/api-guides-tutorials/document-digitization/overview)
- [Start asynchronous job](https://docs.sarvam.ai/api-reference-docs/document-intelligence/start)
- [Job status and page metrics](https://docs.sarvam.ai/api-reference-docs/document-intelligence/get-status)
- [Download result URLs](https://docs.sarvam.ai/api-reference-docs/document-intelligence/get-download-links)

Sarvam Document Digitization does not currently expose the same arbitrary
image-plus-Gemini-prompt contract used by the prompt-compatible providers.
Therefore:

- do not claim that Sarvam received the Gemini prompt;
- do not insert the prompt into an unrelated field;
- do not label Sarvam as a prompt-controlled chat VLM;
- explicitly report its input contract as provider-native document
  digitization.

This is still a valid off-the-shelf OCR comparison, but it is a different
method contract. The report must disclose that difference.

Recommended identity:

```text
provider_id: sarvam
method_id: sarvam_document_e2e
model_id: sarvam-vision
api_key_env: SARVAM_API_KEY
acquisition_kind: async_document_ocr
input_contract: one_source_page_provider_native_digitization_sa_IN
output_adapter_id: sarvam_document_json
```

Do not call it `sarvam_e2e` until the distinction from a future
prompt-compatible Sarvam vision endpoint is unambiguous.

### Page-level acquisition versus batching

The current experimental unit of paid acquisition is one manuscript page.
Prefer one source page per Sarvam job even if the provider supports up to ten
pages per PDF or ZIP.

This preserves:

- one cache fingerprint per source page;
- one remote job ID per source page;
- unambiguous page-to-result mapping;
- page-level failure status;
- page-level cost accounting;
- safe recovery after interruption;
- simple reuse across folds.

If batching is later necessary, introduce a batch acquisition cache as a
separate abstraction. A batch must atomically record:

- the ordered member page IDs;
- every member image hash;
- the uploaded container hash;
- the remote job ID;
- page-index-to-page-ID mapping;
- partial completion and per-page errors;
- cost allocation policy;
- all derived per-page terminal results.

Do not hide a multi-page remote job behind several independent page cache
entries. That can create duplicate jobs and inconsistent terminal state.

### Sarvam asynchronous state journal

Persist transitions before moving to the next remote action:

```text
local_request_created
remote_job_created
upload_requested
upload_completed
job_start_requested
job_started
polling
completed | partially_completed | failed
download_metadata_received
artifact_downloaded
output_adapted
terminal_result_written
```

The journal should include:

- remote job ID;
- provider state;
- timestamps;
- safe response metadata;
- page progress;
- provider error codes;
- page errors;
- downloaded artifact hashes.

Do not persist:

- API keys;
- authorization headers;
- complete presigned URLs;
- signed URL query parameters.

If a process restarts after `remote_job_created`, recover using that job ID.
If it restarts after `job_started`, resume status polling. If it restarts after
completion, request or reuse download metadata and continue adaptation. Never
create a second job automatically for the same fingerprint.

### Sarvam language and request settings

For the current Sanskrit manuscript experiment, pin:

```text
language: sa-IN
output_format: md or html
model: sarvam-vision
```

The structured JSON is the authoritative machine-readable artifact. Markdown
or HTML is useful audit evidence but must not be heuristically parsed into line
geometry when structured fields exist.

Include these settings and their contract version in the fingerprint. Do not
rely on provider defaults, which can change.

Before acquisition, validate:

- supported source format;
- file size;
- page count;
- installed SDK version;
- credential presence;
- rate-limit configuration;
- output-adapter compatibility with a pinned real response fixture.

Sarvam documents a low request-per-minute limit for Document Digitization.
Provider-specific concurrency and polling controls must be explicit. Do not
reuse chat-provider defaults without checking the current official limit.

### Sarvam output eligibility gate

The experiment requires text-line geometry and text, not only document
Markdown.

Before registering Sarvam as a full end-to-end method:

1. Acquire one approved development fixture outside the unit-test suite.
2. Save a sanitized provider-native JSON fixture under experiment tests.
3. Identify the exact documented fields for:

   - page identity or page index;
   - text blocks;
   - text-line objects, if present;
   - coordinates;
   - coordinate order;
   - coordinate units;
   - page dimensions;
   - transcription;
   - reading order;
   - failure metadata.

4. Verify whether geometry is line-level or only region/block-level.
5. Verify coordinate scaling on a known page image.
6. Verify Sanskrit Unicode is preserved without transliteration or
   normalization beyond the experiment's common text normalization.

Do not fabricate line polygons from paragraphs merely to satisfy PAGE-XML.
Do not split Markdown into pseudo-lines. If Sarvam exposes only region boxes or
text without line geometry, it is not yet eligible for the same full
layout-plus-OCR metric table.

In that case, choose explicitly between:

- deferring Sarvam until line geometry is available;
- defining a separate text-only OCR study with metrics and denominators
  designed for text-only output.

Never silently assign GT geometry to Sarvam output. That would change the
method from end-to-end OCR into a GT-layout OCR condition.

### Sarvam PAGE adapter

When the native response supplies adequate line geometry, implement a dedicated
pure adapter:

```python
def sarvam_document_json_to_page(
    payload: Mapping[str, Any],
    *,
    template_page: PageXmlPage,
    expected_page_index: int,
) -> PageXmlPage:
    ...
```

The adapter should:

- validate the pinned provider schema;
- select exactly one source page;
- map documented coordinates into page pixels;
- repair polygons only through the accepted common experiment policy;
- preserve provider line ordering metadata where useful;
- assign deterministic line and region IDs;
- preserve Unicode text;
- return a `PageXmlPage`;
- perform no network or filesystem I/O.

The cache layer should write the returned page using the common PAGE writer.
The adapter must be independently unit-testable from a static fixture.

If the provider gives rectangles, use rectangles honestly. Do not synthesize
curved polygons. If the provider gives polygons, preserve them after the common
validation and clipping policy.

### Sarvam cost reporting

Sarvam Document Digitization may bill per page rather than per token. Extend
the normalized usage model instead of pretending page charges are token
charges.

A future provider-neutral usage record should support both:

```text
input_tokens
output_tokens
total_tokens
```

and:

```text
billed_pages
provider_charge
charge_currency
pricing_unit
pricing_version_or_note
```

The report should display:

- unique acquired pages;
- successful and failed pages;
- remote job attempts;
- resumed jobs;
- billed pages;
- known provider charge;
- missing billing metadata.

Do not convert INR to USD using a live exchange rate inside report generation.
If cross-currency comparison is required, pin the exchange rate, date, and
source in run metadata.

### Sarvam tests

At minimum, add tests proving:

- registry identity and `sa-IN` are pinned;
- the provider-native input contract is reported accurately;
- one source page maps to one remote job;
- the remote job ID is atomically persisted;
- restart resumes an existing job rather than creating a new one;
- polling handles `Completed`, `PartiallyCompleted`, and `Failed`;
- signed URL secrets are not persisted;
- a static provider JSON fixture maps correctly to PAGE-XML;
- page indices and coordinates map to the correct source image;
- absent line geometry fails eligibility rather than using GT geometry;
- partial page failure remains in the evaluation denominator;
- acquisition cost is counted once across repeated fold use;
- `run-methods` performs no Sarvam import, credential load, or API call.

## Registry and Reporting Rules

Adding a provider should require one registry entry plus its driver and output
adapter. Avoid adding parallel hard-coded lists in:

- CLI choices;
- method ordering;
- report labels;
- off-the-shelf table definitions;
- pricing logic;
- runtime dispatch.

Where such lists still exist, refactor them to derive from one immutable
provider registry.

Every provider row should disclose:

```text
provider
exact model
input contract
prompt contract or "not applicable"
output adapter
test-page contract
cache schema
request-contract version
```

Prompt-compatible providers may be compared under the shared Gemini prompt.
Provider-native OCR services such as Sarvam must be labeled with their true
contract.

## Required End-to-End Validation

Before any real paid acquisition:

1. Run all experiment unit tests in `gnn_layout`.
2. Verify no tracked file under `app/` changed.
3. Verify provider credentials are present without printing them.
4. Verify SDK imports before creating cache page directories.
5. Run the driver and output adapter against fakes and static fixtures.
6. Confirm the cache path is new or completely compatible.
7. Confirm the discovered page sets for all manuscripts.
8. Confirm model, language, prompt, and request settings in the manifest.
9. Confirm retries are experiment-owned.
10. Confirm the fold runner rejects a missing or mismatched cache before doing
    local work.

After an approved small paid smoke acquisition:

1. Inspect request and result artifacts for secrets.
2. Inspect raw and normalized output.
3. Render or inspect predicted PAGE geometry.
4. Confirm Unicode Sanskrit text is preserved.
5. Confirm terminal rerun performs zero provider calls.
6. Confirm fold materialization copies only selected test pages.
7. Confirm page-level metrics include failures.
8. Confirm provider usage is counted once rather than once per fold.

Only then acquire every page of every manuscript.

## Anti-Patterns

Do not:

- call a provider from `run-methods`;
- acquire only fold test pages;
- regenerate a provider prediction for each fold;
- silently skip failed pages;
- reuse a cache after model or request settings change;
- let SDK retries bypass attempt accounting;
- register text-only DeepSeek behind another provider's vision proxy;
- claim Sarvam used the Gemini prompt when using Document Digitization;
- parse Sarvam Markdown into invented line polygons;
- use GT geometry to make a provider look end-to-end;
- store credentials or signed URLs;
- couple provider code to a particular manuscript name;
- modify production `app` code for an experiment-only integration;
- execute real provider calls from unit tests.

## Definition of Done

A DeepSeek or Sarvam integration is complete only when:

- its method contract is scientifically accurate;
- its exact model and request settings are pinned;
- all pages are acquired independently of folds;
- paid calls are safely resumable or never silently repeated;
- raw provider evidence and normalized PAGE-XML are retained;
- failures remain in the metric denominator;
- output geometry is provider-produced and valid for the reported condition;
- `run-methods` is fully offline;
- unique acquisition usage is reported once;
- all existing and new tests pass;
- no production `app` file is changed;
- the experiment README and CLI examples are updated after the active
  acquisition run has finished.
