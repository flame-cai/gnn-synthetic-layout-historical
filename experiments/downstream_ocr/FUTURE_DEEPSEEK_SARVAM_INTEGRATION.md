


In for these experiment results, I want you to change how the metric TextEdit is being calculated, and then regenerate the report for:
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\app\tests\logs\ocr_5fold_new
Everything else should remain unchanged. We just want to precisely swap and change how TextEdit is being calculated, and then regenerate the report for all competing methods.

For all competing methods, we already have their predictions in the standard PAGE-XML format. The Ground Truths are also in the standard PAGE-XML format. So for each page in each manuscript, we will have a prediction and a Ground truth, both in PAGE-XML format.

Treat every <TextLine><TextEquiv><Unicode> value as one input text_block, ignore coordinates, baselines, region labels, and region geometry.

What the PAGE-XML adapter should extract:

From this:

<TextRegion id="region_0" custom="textbox_label_2">
    <Coords points="..." />

    <TextLine id="region_0_line_0" custom="structure_line_id_0">
        <Coords points="..." />
        <Baseline points="..." />

        <TextEquiv>
            <Unicode>शिक्षाकल्पो...</Unicode>
        </TextEquiv>
    </TextLine>
</TextRegion>

extract only:

शिक्षाकल्पो...

Ignore:

- <TextRegion> IDs;
- <TextRegion custom="...">;
- region coordinates;
- line coordinates;
- baselines;
- image width and height;
- all geometric overlap or distance.

You will still traverse through <TextRegion> to find its descendant <TextLine> elements, but you will not use any region attribute or geometry in scoring.

Completely remove the existing TextEdit metric calculation. We want a fresh new implementation.
Refer to the code in the in the official documentation and reuse it as much as possible to keep the metric calculation fair:
https://github.com/opendatalab/OmniDocBench/tree/v1_5


The order of the TextLines in the Page-XML do not matter so let's go with simple_match instead of quick_match.

For each page:

G={g1​,…,gn​}

is the set of GT <TextLine> transcriptions, and

P={p1​,…,pm​}

is the set of predicted <TextLine> transcriptions.

OmniDocBench’s simple_match:

normalizes every GT and prediction string;
computes every pairwise normalized edit distance;
uses scipy.optimize.linear_sum_assignment—the Hungarian algorithm—to find a minimum-cost one-to-one assignment;
treats unmatched GT lines as deletions;
treats unmatched predicted lines as insertions.

It does not constrain a match by XML sequence position or geometry.

Therefore:

GT XML order:   A, B, C
Pred XML order: C, A, B

can still yield perfect matching:

A ↔ A
B ↔ B
C ↔ C
Custom adapter responsibility

The custom adapter should only perform:

PAGE-XML
    ↓
Extract one Unicode string per TextLine
    ↓
Construct OmniDocBench-compatible text items

Everything after that should remain the official OmniDocBench v1.5 implementation:

get_gt_pred_lines
    ↓
textblock2unicode + clean_string
    ↓
compute_edit_distance_matrix_new
    ↓
match_gt2pred_simple
    ↓
call_Edit_dist

We will reuse the code to calculate the official pairwise cost.


Adapter output
Ground-truth item
{
    "category_type": "text_block",
    "text": unicode_text,
    "attribute": {},
    "position": [index, index],
    "source_id": textline_id,
}
Prediction item
{
    "category_type": "text_all",
    "content": unicode_text,
    "position": [index, index],
    "source_id": textline_id,
}

The position value is retained only because the OmniDocBench data structures expect it in some output and unmatched-item paths. Under simple_match, it does not influence the cost matrix or Hungarian assignment. You can safely assign the XML traversal index:

"position": [index, index]

This does not mean the metric uses XML order.

PAGE-XML parser
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS = {"pc": PAGE_NS}


def extract_unicode(text_line: ET.Element) -> str | None:
    """
    Extract one transcription from TextLine/TextEquiv/Unicode.

    When several TextEquiv alternatives exist, prefer the smallest numeric
    TextEquiv@index. A missing index is treated as zero.
    """
    candidates: list[tuple[int, str]] = []

    for text_equiv in text_line.findall("./pc:TextEquiv", NS):
        unicode_element = text_equiv.find("./pc:Unicode", NS)

        if unicode_element is None or unicode_element.text is None:
            continue

        try:
            index = int(text_equiv.get("index", "0"))
        except ValueError:
            index = 0

        candidates.append((index, unicode_element.text))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def parse_pagexml(
    path: str | Path,
    *,
    ground_truth: bool,
) -> tuple[str, list[dict]]:
    """
    Convert PAGE-XML TextLines into OmniDocBench-compatible text items.

    Coords, Baseline, TextRegion attributes, region IDs and geometry are ignored.
    """
    path = Path(path)

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid PAGE-XML file: {path}") from exc

    page = root.find("./pc:Page", NS)

    if page is None:
        raise ValueError(f"No PAGE Page element found in {path}")

    image_name = page.get("imageFilename") or path.stem
    items: list[dict] = []

    # XML order is used only to create stable bookkeeping indices.
    # simple_match does not use it as a matching constraint.
    for text_line in page.findall(".//pc:TextLine", NS):
        text = extract_unicode(text_line)

        if text is None or text == "":
            continue

        index = len(items)
        source_id = text_line.get("id", f"line_{index}")

        if ground_truth:
            item = {
                "category_type": "text_block",
                "text": text,
                "attribute": {},
                "position": [index, index],
                "source_id": source_id,
            }
        else:
            item = {
                "category_type": "text_all",
                "content": text,
                "position": [index, index],
                "source_id": source_id,
            }

        items.append(item)

    return image_name, items

Using:

page.findall(".//pc:TextLine", NS)

means that <TextRegion> serves only as an XML container through which the parser descends. Its attributes, label, coordinates and grouping are not used.

Calling simple_match
from utils.match import match_gt2pred_simple


image_name, gt_items = parse_pagexml(
    gt_xml_path,
    ground_truth=True,
)

_, pred_items = parse_pagexml(
    pred_xml_path,
    ground_truth=False,
)

matches, unmatched_table_predictions = match_gt2pred_simple(
    gt_items,
    pred_items,
    "text",
    image_name,
)

assert unmatched_table_predictions is None

Your configuration should use:

end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist

  dataset:
    dataset_name: pagexml2pagexml_dataset

    ground_truth:
      data_path: /path/to/gt_xml

    prediction:
      data_path: /path/to/pred_xml

    match_method: simple_match
What must remain unchanged

Keep the following official v1.5 functions unchanged:

utils.data_preprocess.textblock2unicode
utils.data_preprocess.clean_string
utils.match.get_gt_pred_lines
utils.match.compute_edit_distance_matrix_new
utils.match.match_gt2pred_simple
metrics.cal_metric.call_Edit_dist

The final evaluator recomputes raw Levenshtein counts for every matched or unmatched record, calculates each page’s ratio, and reports the mean of those page-level values as ALL_page_avg.

What this metric now measures

This adaptation measures:

character recognition quality;
whether GT and predicted lines can be paired by textual similarity;
missing text lines;
extra text lines.

It deliberately does not measure:

reading order;
XML order;
physical location;
baseline accuracy;
region grouping;
region classification;
whether a predicted line belongs to the correct <TextRegion>.

It also does not tolerate line splitting and merging:

GT:
ABC DEF

Prediction:
ABC
DEF

Because the prediction contains two atomic units and GT contains one, simple_match can match only one predicted line to the GT line. The other becomes an unmatched insertion. This is appropriate when each <TextLine> should be treated as atomic.

One small v1.5 code issue

The official match_gt2pred_simple contains checks resembling:

if pred_idx

A valid prediction index of 0 is false in Python. This can cause the diagnostic fields pred_category_type or pred_position for prediction zero to be left blank, even though the text itself is matched correctly.

For robustness, patch only those checks:

if pred_idx != ""

rather than:

if pred_idx

This does not alter the assignment or TextEdit score; it fixes metadata associated with prediction index zero.

A precise name for the adapted score would be:

Unordered PAGE-XML TextLine TextEdit, using OmniDocBench v1.5 simple_match

That name makes clear that each PAGE <TextLine> is atomic and that order and geometry are intentionally ignored.

# Implement unordered PAGE-XML TextLine TextEdit using OmniDocBench v1.5

## Objective

Implement a TextEdit-style metric for a historical dataset in which both ground truth and predictions are stored as PAGE-XML using the 2013 namespace:

```text
http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15
```

Reuse as much code as possible from the official OmniDocBench v1.5 implementation.

The only custom component should be an adapter that converts PAGE-XML files into the item structures expected by OmniDocBench. Do not reimplement text normalization, edit-distance calculation, Hungarian matching, unmatched-item handling, or metric aggregation.

## Official references

Use the following official sources:

* [OmniDocBench v1.5 repository](https://github.com/opendatalab/OmniDocBench/tree/v1_5)
* [OmniDocBench paper on arXiv](https://arxiv.org/abs/2412.07626)
* [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf)
* [Official `utils/match.py`](https://github.com/opendatalab/OmniDocBench/blob/v1_5/utils/match.py)
* [Official `utils/data_preprocess.py`](https://github.com/opendatalab/OmniDocBench/blob/v1_5/utils/data_preprocess.py)
* [Official `metrics/cal_metric.py`](https://github.com/opendatalab/OmniDocBench/blob/v1_5/metrics/cal_metric.py)
* [Official `dataset/md2md_dataset.py`](https://github.com/opendatalab/OmniDocBench/blob/v1_5/dataset/md2md_dataset.py)
* [Official dataset registry](https://github.com/opendatalab/OmniDocBench/blob/v1_5/registry/registry.py)
* [Official configuration directory](https://github.com/opendatalab/OmniDocBench/tree/v1_5/configs)
* [Official CLI entry point](https://github.com/opendatalab/OmniDocBench/blob/v1_5/pdf_validation.py)
* [Official Apache-2.0 license](https://github.com/opendatalab/OmniDocBench/blob/v1_5/LICENSE)

Pin the dependency to the `v1_5` branch and record the exact Git commit hash used.

## Metric definition for this dataset

Treat every PAGE-XML `<TextLine>` as one atomic text unit.

For each `<TextLine>`, extract only:

```xml
<TextLine>
    <TextEquiv>
        <Unicode>...</Unicode>
    </TextEquiv>
</TextLine>
```

Each extracted `<Unicode>` value becomes one initial text item.

Use OmniDocBench:

```text
simple_match
```

Do not use:

```text
quick_match
```

The metric must treat GT and prediction TextLines as unordered sets. Their order in the XML document must not constrain matching.

The final matching must use the official global one-to-one assignment implemented by:

```python
utils.match.match_gt2pred_simple
```

This function uses the official normalized edit-distance matrix and SciPy's Hungarian assignment through:

```python
scipy.optimize.linear_sum_assignment
```

## Information that must be ignored

Do not use any of the following for matching or scoring:

* `<Coords>`;
* `<Baseline>`;
* `TextRegion` coordinates;
* `TextLine` coordinates;
* image width or height;
* region IDs;
* region labels;
* `TextRegion@custom`;
* `TextLine@custom`;
* region membership;
* XML element order;
* reading order;
* geometric overlap;
* geometric distance.

`<TextRegion>` should be treated only as an XML container through which descendant `<TextLine>` elements are found.

Changing coordinates, baselines, region labels, region IDs, or TextLine order must not change the metric result.

## Custom code boundary

Write a custom dataset adapter, preferably:

```text
dataset/pagexml2pagexml_dataset.py
```

Register it in the OmniDocBench dataset registry under a name such as:

```text
pagexml2pagexml_dataset
```

The adapter should replace only this part of the official pipeline:

```text
Markdown parsing
    ↓
OmniDocBench item lists
```

with:

```text
PAGE-XML parsing
    ↓
OmniDocBench item lists
```

Everything after creation of the item lists must use official OmniDocBench v1.5 code.

## Required adapter behavior

### 1. Pair files by page

Pair GT and prediction XML files by an explicitly documented filename rule, preferably the same filename or the same filename stem.

Do not silently skip a page when its prediction XML file is missing.

For a missing prediction page, pass:

```python
pred_items = []
```

to the official matcher so the missing content is penalized.

### 2. Parse the PAGE namespace correctly

Use namespace-aware XML parsing for:

```text
http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15
```

For example:

```python
PAGE_NS = {
    "pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
}
```

Locate TextLines using a namespace-aware path such as:

```python
page.findall(".//pc:TextLine", PAGE_NS)
```

Do not parse all `<Unicode>` descendants globally because that could accidentally include a region-level `<TextEquiv>`.

### 3. Select one transcription per TextLine

For each `<TextLine>`, inspect only its direct:

```text
TextLine/TextEquiv/Unicode
```

children.

If multiple `<TextEquiv>` alternatives exist, use a deterministic documented rule:

1. prefer `TextEquiv@index="0"`;
2. otherwise use the smallest numeric `index`;
3. otherwise use the first direct `<TextEquiv>` in document order.

Do not use confidence values to choose the transcription unless this is explicitly added as a separate configurable policy.

### 4. Handle empty text consistently

Treat a missing `<Unicode>` element or an empty/whitespace-only value as no text item.

Apply the same policy to GT and predictions.

Log skipped TextLines with the XML filename and TextLine ID for diagnostics.

### 5. Construct official-compatible item dictionaries

Construct GT items in the form:

```python
{
    "category_type": "text_block",
    "text": unicode_text,
    "attribute": {},
    "position": [bookkeeping_index, bookkeeping_index],
    "source_id": textline_id,
}
```

Construct prediction items in the form:

```python
{
    "category_type": "text_all",
    "content": unicode_text,
    "position": [bookkeeping_index, bookkeeping_index],
    "source_id": textline_id,
}
```

`source_id` is custom diagnostic metadata and must not be used for matching.

`position` is bookkeeping metadata required by some OmniDocBench result paths. It must not be included in the cost matrix or used to constrain assignment.

Do not set meaningful reading-order values.

## Official code that must be reused unchanged

### Text extraction from item dictionaries and normalization

Reuse:

```python
utils.match.get_gt_pred_lines
```

This function extracts GT text from:

```python
item["text"]
```

and prediction text from:

```python
item["content"]
```

for text matching.

It also applies the official text normalization:

```python
utils.data_preprocess.textblock2unicode
utils.data_preprocess.clean_string
```

Do not write a replacement normalizer.

Important: OmniDocBench's `clean_string` removes line breaks and characters outside its retained word-character ranges. Preserve this behavior for comparability, even if a different normalization might appear preferable for the historical script.

### Pairwise matching cost

Reuse:

```python
utils.match.compute_edit_distance_matrix_new
```

The official pairwise cost is:

```text
Levenshtein distance / max(length of GT, length of prediction)
```

Do not replace the denominator with GT length or any custom CER denominator.

### Global one-to-one assignment

Reuse:

```python
utils.match.match_gt2pred_simple
```

This must remain the matching entry point.

It constructs the full GT-by-prediction cost matrix and calls:

```python
scipy.optimize.linear_sum_assignment
```

The assignment must not be constrained by:

* XML order;
* PAGE coordinates;
* TextRegion membership;
* line IDs;
* region IDs.

Do not add an order penalty or geometry penalty.

### Missing and extra text units

Reuse the official unmatched-item behavior in:

```python
utils.match.match_gt2pred_simple
```

Do not discard:

* unmatched GT TextLines;
* unmatched predicted TextLines;
* entire missing prediction pages.

Missing GT matches must contribute deletion errors, and extra prediction items must contribute insertion errors according to the official output records and metric aggregation.

### Metric calculation and aggregation

Reuse:

```python
metrics.cal_metric.call_Edit_dist
```

Do not calculate the final score independently.

Report the official:

```text
Edit_dist -> ALL_page_avg
```

result as the primary dataset score.

The official page-level calculation is:

```text
sum of character edit counts on a page
---------------------------------------
sum of max(GT length, prediction length) for the matched records on that page
```

The reported `ALL_page_avg` is the mean of those page-level scores.

Also preserve the official supplementary outputs when available:

```text
edit_whole
edit_sample_avg
```

Do not substitute either of these for `ALL_page_avg`.

## Suggested integration structure

Implement:

```text
dataset/pagexml2pagexml_dataset.py
```

with responsibilities limited to:

1. reading GT XML files;
2. reading prediction XML files;
3. extracting one Unicode transcription per TextLine;
4. constructing OmniDocBench-compatible item dictionaries;
5. invoking `match_gt2pred_simple`;
6. wrapping matched records in the same recognition dataset object used by the official datasets;
7. returning the text samples under the `text_block` category.

Mirror the architecture of:

* [`dataset/md2md_dataset.py`](https://github.com/opendatalab/OmniDocBench/blob/v1_5/dataset/md2md_dataset.py)

but replace:

```python
md_tex_filter(...)
```

with the custom PAGE-XML parser.

Do not copy and modify the contents of `match_gt2pred_simple` into the adapter. Import and call the official function directly.

## Expected high-level flow

```text
GT PAGE-XML ───────┐
                   ├─ custom PAGE-XML adapter
Prediction PAGE-XML┘
                           ↓
               OmniDocBench-compatible items
                           ↓
             get_gt_pred_lines — official
                           ↓
      textblock2unicode + clean_string — official
                           ↓
   compute_edit_distance_matrix_new — official
                           ↓
          match_gt2pred_simple — official
                           ↓
       scipy linear_sum_assignment — official
                           ↓
             call_Edit_dist — official
                           ↓
                   ALL_page_avg
```

## Configuration

Add a configuration similar to:

```yaml
end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist

  dataset:
    dataset_name: pagexml2pagexml_dataset

    ground_truth:
      data_path: /path/to/ground_truth/pagexml

    prediction:
      data_path: /path/to/prediction/pagexml

    match_method: simple_match
```

Reject any configuration that asks this adapter to use `quick_match`, unless support for it is deliberately added later.

## Reproducibility requirements

Record in the result output:

```text
metric_name
OmniDocBench Git commit hash
OmniDocBench branch or tag
adapter version
PAGE namespace
TextEquiv selection policy
empty-text policy
file-pairing policy
normalization functions used
match method
aggregation field
```

Suggested metric name:

```text
Unordered PAGE-XML TextLine TextEdit
```

Suggested full description:

> Unordered PAGE-XML TextLine TextEdit using OmniDocBench v1.5 `simple_match`. Each PAGE `<TextLine>/<TextEquiv>/<Unicode>` transcription is treated as one atomic text item. TextRegion information, XML order, coordinates, baselines, geometry, and reading order are ignored. Text normalization, normalized edit-distance costs, Hungarian assignment, unmatched-item handling, and page-level aggregation are reused from the official OmniDocBench implementation.

Do not describe the result as the official paragraph-level OmniDocBench leaderboard TextEdit score because the initial evaluation granularity here is PAGE TextLine rather than OmniDocBench paragraph-level `text_block`.

## Required tests

### 1. Perfect match

GT and prediction contain the same TextLine strings.

Expected:

```text
ALL_page_avg = 0
```

### 2. XML-order invariance

GT:

```text
A, B, C
```

Prediction:

```text
C, A, B
```

Expected:

```text
score = 0
```

when all strings are otherwise identical and distinct.

### 3. Coordinate invariance

Change every `<Coords>` and `<Baseline>` value while preserving Unicode text.

Expected:

```text
score unchanged
```

### 4. Region invariance

Move TextLines between TextRegions and change all TextRegion IDs and `custom` attributes while preserving Unicode text.

Expected:

```text
score unchanged
```

### 5. Atomic-line behavior

GT:

```text
ABC DEF
```

Prediction:

```text
ABC
DEF
```

Expected:

* no merge of the two prediction TextLines;
* one prediction may match the GT line;
* the other remains unmatched;
* the result is not a perfect score.

This confirms that `simple_match`, not adjacency merging, is being used.

### 6. Missing GT match

A GT TextLine has no corresponding prediction.

Expected:

* the GT item remains in the matched-record output with an empty prediction;
* it contributes a deletion penalty.

### 7. Extra prediction

A predicted TextLine has no corresponding GT TextLine.

Expected:

* the prediction remains in the matched-record output as extra content;
* it contributes an insertion penalty.

### 8. Missing prediction page

A GT XML file exists but the corresponding prediction file does not.

Expected:

* the page is not skipped;
* the adapter passes an empty prediction list;
* all GT text is penalized as missing.

### 9. Duplicate text

GT and prediction contain repeated identical or nearly identical lines.

Expected:

* matching remains one-to-one;
* one prediction cannot satisfy multiple GT items.

### 10. Multiple TextEquiv alternatives

Create a TextLine with several `<TextEquiv>` entries.

Expected:

* the documented deterministic selection policy is followed;
* the behavior is identical in GT and prediction parsing.

### 11. Original-code equivalence

Create an in-memory set of GT and prediction item dictionaries and compare:

```python
match_gt2pred_simple(adapter_items...)
```

against the adapter's matched output.

Expected:

```text
identical normalized strings, assignment and edit records
```

apart from explicitly added diagnostic metadata such as `source_id`.

## Minimal-change principle

The implementation should contain custom logic only where the official OmniDocBench code assumes Markdown or OmniDocBench JSON input.

Do not fork, copy, or rewrite the metric internals unless strictly necessary.

Preferred imports include:

```python
from utils.match import match_gt2pred_simple
from metrics.cal_metric import call_Edit_dist
```

and the official matcher should continue importing and using:

```python
get_gt_pred_lines
compute_edit_distance_matrix_new
textblock2unicode
clean_string
linear_sum_assignment
```

Any unavoidable patch to official code must be:

1. minimal;
2. documented;
3. covered by a regression test;
4. shown not to change the numerical score on official-compatible inputs.

Retain the OmniDocBench Apache-2.0 license and required notices for reused code.

One implementation detail worth testing carefully: 
v1.5 uses truth-value checks on pred_idx when filling some diagnostic metadata, so prediction index 0 can lose its category or position metadata even though its text assignment and numerical edit score are still calculated.



























We much implement textedit exactly, but treat each text-line predicted by Sarvam and also  (Gemini, OpenAI, claude) as a text_block, after applying the respective appropriate adapters for both Sarvam and (Gemini, OpenAI, claude). But everything else needs to be exactly implemented.

Then implement sarvam. It does predicted with text-line level granularity. So we need an adapter which will parse the predicted text-lines in HTML as text-blocks for a fair evaluation with Gemini and OpenAI.

- Sarvam gives text-line level text (the textedit matching can help Sarvam)
- GT has text-line level text 
- Annotation tool has text-line level text (the textedit cannot help)
- Gemini, Openai, Claude also has text-line level text. (the textedit cannot help)

################

Thus we can use the TextEdit Metric from OmniDocBench, but should consider each detected text-line as a paragraph when calculating the metric.
TextEdit metric does not depend on the text-line locations or anything spatial.

The requirements are:

- P(1) and P(2) must be adjacent to each other so they can be merged.
- Their internal order must be correct.
- The merged text must be sufficiently similar to GT(122)

There is no requirement in the original OmniDocBench TextEdit matching that prediction index 1 must match a GT index near 1. The matching searches for textually similar GT–prediction pairs and allows adjacent paragraphs to be merged or split.

so being predictions being adjacent to each other help the matching? but is this matching like almost being lenient? and if the predictions are already "good" in the sense that they don't need to be split or merged.. then the matching (and the prediction order) should not matter?

Your conclusion is therefore correct under these conditions:
- predictions already form good one-to-one paragraphs;
- paragraphs are sufficiently distinct to match unambiguously;
- there are no major missing or duplicate blocks.
###########



In OmniDocBench’s evaluation terminology, the textual units evaluated by TextEdit are grouped under the category:

text_block

The configuration uses:

text_block:  # Configuration for text paragraphs
  metric:
    - Edit_dist

For predictions, the model normally outputs page-level Markdown. OmniDocBench parses and segments that Markdown—primarily using blank lines or double line breaks—to create paragraph-like units that are evaluated as text_block items.

The approximate pipeline is:

Prediction Markdown
        ↓
Markdown parsing and paragraph segmentation
        ↓
Sequence of text_block evaluation items
        ↓
simple_match or quick_match
        ↓
TextEdit

The important nuance is that text_block is an evaluation category, not a guarantee that every unit is a true semantic paragraph.

A predicted text_block may correspond to:

a text-line in a historical manuscript page (which is prepared by an adapter to convert predicted Sarvam and (Gemini, Openai)) outputs.


###################
Saravam Vision does not localize. It output in HTML format, without the locations of the text-lines.
It however distinguishes between text-lines: which can be deduced using </p> and </br>.

<p class="paragraph">हृत्तिको द्यात<br/>
प्राग्भवति=१</p>
<p class="paragraph">ब्राह्मणः=५<br/>
विप्रादयः=५<br/>
स्वप्राकास्पस्थले=५<br/>
एजनसर्वनेत्वा<br/>
=३</p>


# Example Input:
from sarvamai import SarvamAI

client = SarvamAI(
    api_subscription_key="YOUR_API_KEY",
)

## Create a document intelligence job
job = client.document_intelligence.create_job(
    job_parameters=dict(
        language="sa-IN",
        output_format="html",
    ),
)
print(f"Job created: {job.job_id}")

## Upload the document (we want to go one page at a time, but in parallel upto 8 pages)
job.upload_files(file_paths=["17.jpg"])
print("File uploaded")

## Start processing
job.start()

## Wait for completion (polls /status)
status = job.wait_until_complete()
print(f"Job completed with state: {status.job_state}")

## Download outputs (writes to ./outputs)
job.download_outputs(output_dir="./outputs")
print("Outputs saved to ./outputs")


# Example Output:
<!DOCTYPE html>
<html lang="sa-IN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
body {
    padding: 0 !important;
    margin: 0 !important;
    background-color: white !important;
    width: 100% !important;
    font-family: serif;
}

.page-body-container {
    background-color: white;
    width: 210mm;
    margin: 0 auto;
    padding: 20mm 15mm;
    box-sizing: border-box;
    min-height: 100vh;
}

p.paragraph, .paragraph, .block, .quote, .ordered-list, .unordered-list,
.advertisement, .answer, .contact-info, .index, .options,
.reference, .unknown, .sidebar, .footnote {
    text-align: justify !important;
    text-align-last: left !important;
    text-justify: inter-word !important;
    font-size: inherit !important;
    line-height: 1.3 !important;
    margin: 0 0 12pt 0 !important;
    text-indent: 18pt !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
}

.quote, .ordered-list, .unordered-list, .sidebar, .footnote {
    text-indent: 0 !important;
}

.multi-column-row { font-size: 12pt; }
.multi-column-row.cols-1 { font-size: 12pt; }
.multi-column-row.cols-2 { font-size: 11pt; gap: 18pt; }
.multi-column-row.cols-3 { font-size: 10pt; gap: 16pt; }
.multi-column-row.cols-4 { font-size: 9pt; gap: 12pt; }

.page-body-container > p.paragraph,
.page-body-container > .paragraph {
    font-size: 12pt !important;
}

h1.chapter-title {
    font-size: 30pt !important;
    line-height: 1.1 !important;
    margin: 0 0 12pt 0 !important;
    text-align: center !important;
    font-weight: bold;
}

h2.section-title, h2.headline {
    font-size: 18pt !important;
    line-height: 1.1 !important;
    margin: 12pt 0 8pt 0 !important;
    text-align: center !important;
    font-weight: bold;
}

h3.sub-section-title, h3.sub-headline,
h4.subsub-section-title, h4.subsub-headline {
    font-size: 13pt !important;
    line-height: 1.25 !important;
    margin: 12pt 0 4pt 0 !important;
    text-align: center !important;
    font-weight: bold;
}

.multi-column-row {
    width: 100% !important;
    margin: 12pt 0 !important;
    display: grid;
}

.column-block {
    text-align: left !important;
    text-align-last: left !important;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    min-width: 0;
    overflow-wrap: break-word;
    width: 100%;
}

.column-block p {
    text-indent: 0 !important;
    margin-bottom: 8pt !important;
    font-size: inherit !important;
}

.column-block p br {
    display: none;
}

.centered-block {
    width: 100%;
    display: flex;
    justify-content: center;
    margin: 12pt 0;
}

.centered-block > * {
    text-align: center !important;
    font-size: 13pt !important;
    max-width: 100% !important;
    margin: 0 auto !important;
    text-indent: 0 !important;
}

.table {
    max-width: 100%;
    overflow-x: auto;
    margin: 18pt 0;
}

.table table {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    border-collapse: collapse;
    border: 1.5pt solid #333;
    table-layout: auto;
    font-size: 11pt;
    line-height: 1.3;
    font-variant-numeric: tabular-nums;
}

.table table td,
.table table th {
    word-break: normal;
    overflow-wrap: break-word;
    hyphens: manual;
    white-space: normal;
    vertical-align: top;
    padding: 6pt 8pt;
    font-size: inherit;
    border: 0.5pt solid #333;
    text-align: left;
}

.table table th {
    background-color: #f2f2f2;
    font-weight: bold;
    text-align: center;
    border-bottom: 1pt solid #333;
}

.table table td.numeric,
.table table th.numeric {
    text-align: right;
}

.table table td.nowrap,
.table table th.nowrap {
    white-space: nowrap;
}

/* Subtle zebra striping improves scannability on multi-row receipts. */
.table table tbody tr:nth-child(even) td {
    background-color: #fafafa;
}

/* Emphasise totals rows. A ``tfoot`` is the semantic ideal, but most OCR
   outputs use a bold-labelled final ``tr``, so we cover both forms. */
.table table tfoot td,
.table table tfoot th,
.table table tr:last-child td:has(b),
.table table tr:last-child td:has(strong) {
    border-top: 1pt solid #333;
    font-weight: bold;
    background-color: #f2f2f2;
}

.column-block .table {
    max-width: 100%;
    overflow-x: auto;
    margin: 12pt 0;
}

.column-block .table table {
    width: 100% !important;
    table-layout: auto;
    margin: 0;
    border-collapse: collapse;
    border: 1.5pt solid #333;
    font-variant-numeric: tabular-nums;
}

.column-block .table table td,
.column-block .table table th {
    word-break: normal;
    overflow-wrap: break-word;
    hyphens: manual;
    white-space: normal;
    vertical-align: top;
    padding: 6pt;
    font-size: 10pt;
    border: 0.5pt solid #333;
    text-align: left;
}

.column-block .table table th {
    background-color: #f2f2f2;
    font-weight: bold;
    text-align: center;
    border-bottom: 1pt solid #333;
}

.column-block .table table td.numeric,
.column-block .table table th.numeric {
    text-align: right;
}

.column-block .table table td.nowrap,
.column-block .table table th.nowrap {
    white-space: nowrap;
}

.column-block .table table tbody tr:nth-child(even) td {
    background-color: #fafafa;
}

.column-block .table table tfoot td,
.column-block .table table tfoot th,
.column-block .table table tr:last-child td:has(b),
.column-block .table table tr:last-child td:has(strong) {
    border-top: 1pt solid #333;
    font-weight: bold;
    background-color: #f2f2f2;
}

.quote {
    font-size: 13pt !important;
    margin: 0 0 0 0 !important;
    padding: 0 20px !important;
    font-style: italic;
    border-left: 3px solid #ccc;
    text-align: justify !important;
    text-indent: 0 !important;
}

.ordered-list, .unordered-list {
    font-size: inherit !important;
    margin: 0 0 0 0 !important;
    padding-left: 40px !important;
    text-align: justify !important;
    text-indent: 0 !important;
}

.sub-ordered-list, .sub-unordered-list {
    font-size: inherit !important;
    margin: 0 0 0 0 !important;
    padding-left: 60px !important;
    text-align: justify !important;
    text-indent: 0 !important;
}

.subsub-ordered-list, .subsub-unordered-list {
    font-size: inherit !important;
    margin: 0 0 0 0 !important;
    padding-left: 80px !important;
    text-align: justify !important;
    text-indent: 0 !important;
}

.image, .chart, .diagram, .photograph {
    margin: 16px auto;
    text-align: center;
    max-width: 100% !important;
}

.image img, .chart img, .diagram img, .photograph img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
    border: 1px solid #ddd;
}

.chart {
    border: 2px solid #4a90d9;
    padding: 8px;
    background-color: #f8fafc;
}

.diagram {
    border: 2px solid #6b7280;
    padding: 8px;
    background-color: #f9fafb;
}

.photograph {
    border: 2px solid #059669;
    padding: 4px;
    background-color: #f0fdf4;
}

.image-caption {
    font-style: italic;
    margin: 8px auto;
    text-align: center;
    max-width: 100% !important;
}

.header, .footer {
    font-size: 13pt !important;
    margin: 12px auto;
    text-align: center;
    max-width: 100% !important;
}

.sidebar {
    font-size: inherit !important;
    margin: 12px auto;
    padding: 10px;
    background-color: #f5f5f5;
    text-align: justify !important;
    text-indent: 0 !important;
}

.footnote {
    font-size: inherit !important;
    margin: 8px auto;
    padding-left: 20px;
    text-align: justify !important;
    text-indent: 0 !important;
}

hr.footnote-separator {
    border: none;
    border-top: 1px solid #999;
    width: 40%;
    margin: 16px 0 8px 0;
}

a.footnote-ref {
    color: inherit;
    text-decoration: none;
}

a.footnote-ref:hover {
    text-decoration: underline;
}

.advertisement, .answer, .contact-info, .index, .options,
.reference, .unknown {
    font-size: inherit !important;
    margin: 0 0 12pt 0 !important;
    text-align: justify !important;
}

.author, .dateline, .flag, .formula, .jumpline,
.page-number, .folio, .website-link,
.first-level-question, .second-level-question, .third-level-question {
    font-size: inherit !important;
    margin: 12px auto;
    text-align: center;
    max-width: 100% !important;
    text-indent: 0 !important;
}
    </style>
</head>
<body>
<div class="page-body-container">
<p class="paragraph">स्वस्मात्पंचमः परैश कन्यायुक्तव्यतिरिक्तचत्वारोशशयः २।२।५।९=२<br/>
उर्येषां=२ अन्यवर्गयुक्तः=२ वल</p>
<p class="folio">बलवंतोभवंति=</p>
<div class="multi-column-row cols-3" style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));">
<div class="column-block" style="grid-column: 1;">
<p class="paragraph">ब्राह्मणः=५<br/>
विप्रादयः=५<br/>
स्वप्राकास्पस्थले=५<br/>
एजनसर्वनेत्वा<br/>
=३</p>
<p class="paragraph">हृत्तिको द्यात<br/>
प्राग्भवति=१</p>
</div>
<div class="column-block" style="grid-column: 2;">
<p class="paragraph">क्रियेघटा९ कोर्पन्ट उजो१२ याम्यतोमध्यैनैवसंत्यथेंद्रककुभोवर्गाःस्पुरोजे<br/>
स्विनः ८६ अष्टावुप्रमुखाःस्वपंचमेपरादिघ्नःस्ववगेन्पियुक् तष्टःकाकिलिकोग<br/>
जै८मिथइमायस्पाधिकासौर्थदेः श्वेतारक्तकपीतकलहवसुधा‘स्वादुःकडुस्तिक्तका<br/>
काषायाघतशोणितान्नमदिरागंधासुभाविप्रतः ८) सोम्पादिल्लवभूतलैविरचये<br/>
हिप्रादिकोऽग्योखिले नान्येषानियमोचयत्रनिखिलाः कुयुर्गहं ह्तिस्थिरे समप्रह्मक<br/>
तो‘मुखात्प्रथमतोवगादिवलेहिमश्वेतहिग्गतमादिशेतुहयपैःशल्पैसुधा’मध्वितः ८८<br/>
स्वभ्रंहस्तमितरखनेदिहजलंएनिशास्येन्यसेत्प्रातईष्टजलंस्यलसर्दजलेंमध्यत्वंस<br/>
त्स्फारितं‘ज्ञात्वैवनि खनेद्हाधि,कभुवनत्वाजलांतस्तरोयावहापुरुषस्ततःकपिशि,<br/>
रस्तुल्याएमभिःपरयेत्,८९ प्राकुसाध्यो‘ज,यिनीस्यलाद्यमदिदित्वाष्ट्रानिलाभ्यंतरा<br/>
सोम्पेऽतौग्नुदर्यादुदकेध्रुवमुखादिग्मूढकेस्यामितिः ९० गेहंमाधवपोषफालानन<br/>
उज्जायेनीमारपदक्षिणाशायांस्थितेनैनरैःचित्रास्वात्योःउदयेमध्वभागेप्राकृसाध्या=</p>
</div>
<div class="column-block" style="grid-column: 3;">
<h2 class="section-title">सर्वदिक्सकाश<br/>
तत्त्वज्ञानम्</h2>
<p class="paragraph">विप्रादितः कमात<br/>
उत्तरादिनतम्<br/>
मोविप्रादिको गहैं<br/>
विरचयेते = ४<br/>
लोहास्यगारादि=६</p>
<p class="paragraph">यावज्जलंतिःस<br/>
रतिश्चथवाश्वेत<br/>
पीतादिउत्तरः<br/>
मृत्तिकापुटंपुरुष<br/>
प्रमाणावानि वने<br/>
रा=२<br/>
त=३</p>
</div>
</div>
</div>
</body>
</html>













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

Do not Batch. Go one page at a time, with upto 8 API calls in parallel.
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
