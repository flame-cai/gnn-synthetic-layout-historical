
# Clean Layout-Grounded Gemini OCR Ablation Plan

## Summary
Move Gemini OCR toward a cleaner layout-only ablation by making the production app’s layout-grounded Gemini path use visual Coords-based spotlighting plus native Gemini `response_schema`, then keep the experiment’s `gemini_gt_layout` method calling that same production path. The GUI route contracts, OCR registry behavior, active-learning save contract, and local OCR path remain unchanged.

## Key Changes
- Add production Gemini scaffolding in a small internal helper module under `app/` for:
  - `GeminiLineGeometry` records with real PAGE line id, simple badge id, baseline points, Coords polygon points, and drawing thickness.
  - Native `types.Schema` builders for layout-grounded badge OCR responses.
  - Spotlight image generation from PAGE `Coords`, falling back to baseline strokes when `Coords` are missing or invalid.
  - Strict JSON parsing for badge responses, mapping badge ids back to original PAGE `structure_line_id_*` values.
- Rewrite `_run_gemini_recognition_internal` to:
  - Build line geometry from saved PAGE-XML, including `Coords` in `lines_geometry`.
  - Generate one spotlighted RGBA image before the thread pool.
  - Send the spotlighted image plus a concise badge prompt to Gemini.
  - Use `response_mime_type="application/json"` and `response_schema=...`.
  - Preserve the existing return shape: `{"text": {line_id: text}, "confidences": ...}` or `{"error": ..., "errorCode": ...}`.
  - Preserve `ensemble_text_samples` by mapping badge ids to line ids inside each sample worker before ensemble aggregation.
- Update experiment Gemini methods:
  - Keep `gemini_gt_layout` as the headless production-app call, so it automatically uses the new spotlighted production implementation.
  - Update `vlm_e2e` prompt to the persona-aligned prompt from the brief and add a native PAGE-layout response schema for regions/lines/polygons/text.
  - Simplify `adapter.py` JSON parsing to strict `json.loads`; no markdown fence or substring repair.
  - Round scaled normalized geometry to integer PAGE coordinates before polygon validation/writing.
- Improve Gemini logging and debug hooks:
  - Add `DEBUG_OCR_VISUALS`; when truthy, save spotlight images to `layout_analysis_output/debug/<page>_spotlight.jpg`.
  - On strict JSON parse failure, write raw Gemini text to the same debug folder.
  - Count attempted Gemini requests separately from returned usage metadata; report failed/timeout calls as attempted requests with missing usage instead of appearing free.
- Dependency/documentation updates:
  - Set `google-genai>=2.6.0` in both requirements files because the installed SDK confirms `GenerateContentConfig.response_schema` support.
  - Update downstream OCR README/report wording: the comparison is now schema-aligned and production-code-reused, but `vlm_e2e` still predicts layout while `gemini_gt_layout` receives human/GT layout.

## Test Plan
- Unit tests for production helpers:
  - Spotlight generation darkens non-target areas, preserves target areas, draws cyan outlines and yellow badges.
  - Coords polygons are preferred over baselines; baseline fallback works when Coords are absent.
  - Badge response parser accepts integer and numeric-string ids, maps to real line ids, rejects unknown ids, rejects empty output, and dumps raw text only when debug is enabled.
- Update existing Gemini backend parser tests to strict structured-output behavior instead of fenced/wrapper JSON acceptance.
- Downstream OCR tests:
  - `vlm_json_to_page` still accepts polygon and box payloads.
  - Invalid/empty JSON still raises `AdapterError`.
  - Float normalized polygon values become stable integer PAGE coordinates.
  - Reporting counts failed Gemini attempts as requests with missing usage metadata.
- Validation commands:
  - `conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_backend_unit -v`
  - `conda run -n gnn_layout python -m unittest experiments.downstream_ocr.tests.test_metrics experiments.downstream_ocr.tests.test_reporting -v`
  - Optional no-network visual smoke with a tiny synthetic PAGE-XML fixture and `DEBUG_OCR_VISUALS=1` to inspect the generated spotlight image.

## Assumptions
- The production Gemini behavior should switch to spotlighting by default; no legacy trace-mode feature flag will be kept.
- `/recognize-text`, background `runRecognition`, registry writes, active-learning behavior, and local OCR behavior must remain API-compatible.
- The default Gemini timeout remains controlled by `GEMINI_OCR_TIMEOUT_SECONDS`; this plan does not increase it.
- Geometry schema guarantees structure, not semantic polygon validity; invalid model polygons still fail adapter validation and become failed predictions in the experiment.



More Context:
This plan incorporates the hybrid Coords-based Spotlighting, strict Native Structured Outputs (response_schema), and prompt alignment to ensure the model operates strictly in-distribution.
This serves as a precise blueprint for the agent/developer to execute the updates.
Phase 1: Task 1 - End-to-End OCR (ablation script)
Objective: Transition from out-of-distribution hostile prompting to persona-aligned prompting, and enforce layout geometry strictly through Gemini API schemas.
1. Update Prompt (VLM_END_TO_END_PROMPT in adapter.py):
Action: Replace the current string entirely.
New Content:
code
Text
You are an expert Indologist and Paleographer analyzing historical Sanskrit manuscripts. Your task is to perform a diplomatic transcription of the manuscript image and extract the visual text lines. 
Because historical text is often highly curved or written in circles, you must provide tight polygons (`polygon_2d`) that follow the curvature of each text line exactly. Use as many points as necessary to capture the curve accurately. 
Please extract the regions and text lines exactly according to the requested JSON schema.
2. Implement Native Structured Outputs (run_vlm_end_to_end_gemini):
Action: Construct a google.genai.types.Schema object to replace the text-based JSON schema.
Schema Definition:
type: OBJECT
properties:
status: STRING
regions: ARRAY of OBJECTS
Properties of Region: id (STRING), type (STRING), polygon_2d (ARRAY of ARRAY of NUMBER), box_2d (ARRAY of NUMBER), lines (ARRAY of OBJECTS).
Properties of Line: id (STRING), polygon_2d (ARRAY of ARRAY of NUMBER), box_2d (ARRAY of NUMBER), text (STRING).
Action: In client.models.generate_content, add response_schema=your_schema_object to the GenerateContentConfig. Keep response_mime_type="application/json".
3. Clean Up Adapter Scaffold (adapter.py):
Action: Since the API now guarantees valid JSON, remove the complex regex fallback logic inside parse_json_payload. Reduce it to simply json.loads(text).
Action: In vlm_json_to_page, ensure that if Gemini returns floats for polygon_2d (e.g., 150.5), they are safely cast to int if the downstream PAGE-XML validate_polygon strictly expects integers.
Phase 2: Task 2 - Layout Grounded OCR (app.py)
Objective: Replace OOD text-coordinate arrays with In-Distribution visual Spotlighting (Set-of-Mark prompting) using padded Coords and ID badges.
1. Create the Badge Mapping System (_run_gemini_recognition_internal):
Action: Instead of asking Gemini to output long, complex structure_line_id_xyz, assign a simple integer index (1, 2, 3...) to each line.
Action: Maintain a Python dictionary: badge_to_line_id_map = {1: "12345", 2: "12346"}.
2. Implement PIL Spotlighting Logic (Execute ONCE before threads):
Action: Open pil_img = Image.open(img_path).convert("RGBA").
Action: Create a 70% opacity black overlay (dark_overlay).
Action: Create a blank mask (mask = Image.new("L", pil_img.size, 0)).
Action: Loop through lines_geometry:
Condition A (Valid Coords >= 3 points): Use ImageDraw on the mask. Draw the Coords polygon filled with white (255). To achieve "padding/dilation", also draw the polygon's outline with a thick white line (e.g., width=8).
Condition B (Fallback: Missing or < 3 points): Use Baseline points. Draw a white line on the mask using thickness + padding (e.g., width=thickness + 8, joint='curve').
Action: Composite the image: spotlighted_img = Image.composite(pil_img, Image.alpha_composite(pil_img, dark_overlay), mask).
Action: Draw Visual Borders and Badges on spotlighted_img:
Draw a 2px neon cyan border along the Coords (or Baseline).
At the first coordinate of the line, draw a filled yellow circle (radius ~15px) and draw the integer badge ID (1, 2, 3...) in black text centered inside it.
3. Update Prompt and API Call:
Action: Remove the get_equidistant_points and 0-1000 coordinate normalization loops entirely.
Action: Rewrite prompt_text:
code
Text
You are an expert Indologist transcribing a Sanskrit manuscript. The original image has been modified to spotlight the specific text lines we need transcribed.
Each spotlighted region is surrounded by a cyan border and starts with a numbered yellow badge. 
Transcribe the Devanagari text contained within each cyan border. Output a JSON array linking the badge number (`id`) to the transcribed `text`.
Action: Define a response_schema for Gemini: An ARRAY of OBJECTS containing id (INTEGER or STRING matching the badge) and text (STRING). Add this to GenerateContentConfig.
Action: Pass spotlighted_img (instead of pil_img) to Gemini.
4. Update Response Parsing:
Action: Remove the _json_from_gemini_text and _parse_gemini_transcriptions regex hacks. Use standard json.loads.
Action: Iterate through Gemini's JSON array. Look up the badge id in badge_to_line_id_map to get the real PAGE-XML line_id. Return the mapped dictionary.
Phase 3: Agentic & Manual Debugging Hooks
To ensure safety and facilitate debugging, the implementing agent must add these specific visualizations:
1. Spotlight Image Dumper:
Action: In _run_gemini_recognition_internal, right after spotlighted_img is generated, check for an environment variable DEBUG_OCR_VISUALS.
Action: If True, save spotlighted_img.save(base_path / "layout_analysis_output" / "debug" / f"{page}_spotlight.jpg").
Verification: The human or agent should review this folder to ensure:
The cyan borders don't cut off Sanskrit matras.
Circular text is fully illuminated.
Badges don't overlap critical text.
2. API Payload Logger:
Action: Wrap the json.loads in a try/except block. If it fails (which should be extremely rare with Native Schemas), dump response.text to a text file in the debug folder for inspection.
Phase 4: Upstream & Downstream Impact Checklist
The implementing agent must verify the following during execution:
Upstream - lines_geometry Construction:
Currently, the code builds lines_geometry inside _run_gemini_recognition_internal. The agent must ensure poly_pts (the actual layout Coords) are passed into the lines_geometry dictionary so the PIL drawing loop can access them. Currently, they are extracted but not fully appended to the dictionary used later.
Upstream - Dependency Check:
Ensure google-genai SDK is updated to a version that supports response_schema (v0.1.0+ or the equivalent Google AI Studio library version).
Ensure Pillow is imported properly (Image, ImageDraw, ImageFont for drawing the badge numbers).
Downstream - Ensemble Logic (ensemble_text_samples):
The ensemble function relies on receiving a dictionary of {line_id: text}. Because we are mapping badge_id back to line_id before returning from sample_worker, the ensemble logic will remain 100% intact and unaffected. (The agent must double-check this mapping is executed inside sample_worker).
Performance:
Ensure the PIL Image manipulation (Spotlighting) occurs before the ThreadPoolExecutor. Modifying the image once and passing the object to N workers saves memory and CPU time.

## Relevant links:
The current implemented code for the experiment:
gnn-synthetic-layout-historical\experiments\downstream_ocr

Production "app":
.\app

"yajn" manuscript:
.\app\input_manuscripts\yajn

2-fold, 5-page live run on all competing methods:
C:\tmp\downstream_ocr_report_all_5p2f

Use conda environment gnn_layout

IMPORTANT: Make sure the exiting GT PAGE-XML files remain unchanged. This plan will only change how we do OCR using Gemini (with and without GT Layout Grounding)
