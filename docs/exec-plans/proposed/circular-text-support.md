# PRE-PLAN TODO
1) in this repo, there are two pre-commit checks. can you please tell me what they are and what they check?

2) I want you to make modify the active learning finetuning pre-commit check, which uses the dataset: app\tests\eval_dataset

Right now for this check we assume that the text-line segmentation is perfect, and hence we directly use the ground truth bounding polygons <Coords points=../> from the page-xml files as a starting point to process text-line images to be fed to the OCR model for fine-tuning. We want to change this.
Instead of using the <Coords points=../> I want you to use the ground truth <Baseline points="x1,y1 x2,y2 .."/> as the starting point. We want to process the <Baseline points="x1,y1 x2,y2 .."/> in the page-xml, the heatmap, the original image using a pipeline to actually get the <Coords points=../>. This pipeline should be exactly the same as the pipeline in the application, and the pre-trained only pre-trained only check. Hence we should ideally get the _exact same_ <Coords points=../>, and the eventual processed text-line images for the OCR model after this modification.

So instead of taking the shortcut of directly using the <Coords points=../>, I want the test to take the longcut, and calculate the <Coords points=../> from the <Baseline points="x1,y1 x2,y2 .."/>, the heatmap, the original image, using the pipeline which the application actually uses. We must ensure that the longcut get us the exact same <Coords points=../> which the shortcut used directly.

Hence, ideally, we want to use this condition for getting equivalent outputs (<Coords points=../>, processed text-line images for OCR model) to verify if this update has passed.

Once passed, please clean up stale code, and update the docs.

So after this update, both pre-commit checks will ideally be using equivalent pipelines to get the text-line images, which are to be fed to the OCR model.

IMPORTANT.
To ensure equivalency, you will also need to study how the application pipelines works, especially when the user manually adds or deletes nodes, and how these are processed heuristically. Because sometimes, the heatmaps makes mistakes, so the user has to manually adds and deletes nodes, to get the ground-truth text-line <Baseline points="x1,y1 x2,y2 .."/>. The <Baseline points="x1,y1 x2,y2 .."/>s in the page-xml are already ground-truth, but we don't know which nodes, from which baseline were manually added by the user, and which were deleted. So we need to handle this somehow. Perhaps, we can use a looser notion of equivalency. Please think hard about this before implementation.

In this regards, it is also important to note that <Baseline points="x1,y1 x2,y2 .."/> are essentially processed gnn-format predictions, where all points, having the same label, are joined to form the polyline <Baseline points="x1,y1 x2,y2 .."/>, which denotes a text-line location.



___________

- do you have permission to git commit in this branch only? circular-layout-attempt-2
- first fix the active finetuning test, such that it doesn't directly crop text-lines images from PAGE-XML. Use the pipeline, but ensure the results are equivalent! exact equivalent (direct cropping from- page-xml)
- fix the bug: make sure this error never happens: The pre-commit script ran for about 24 minutes, but the outer conda run wrapper hit the known Windows cp1252 Unicode-printing failure after the subprocess completed. 
- no saving in C:\\temp issue
- understand the current text-lines segmentation strategy and update the prompt.



_______________












This plan will enable us to combine the generative capabilities of LLMs with external verifier metrics to perform step by step evolutionary search in python code space - with the purpose to improve how this application segments text-lines (using the resized_images, heatmaps and the respective predictions of the GNN i.e the <Baseline/> in page-xml). In other words, we want to improve the part of the pipeline which converts the GNN predictions (<Baseline/> in page-xml) to the text-line images which are fed to the downstream OCR model for fine-tuning or inference.

The application already has such a text-line segmentation strategy. We will call this the "benchmark_strategy". However this strategy is flawed and works only for horizontal text-lines (not circular, curved, or vertical lines). To fix these flaws, we will be implementing a new "proposed_strategy". The goal is that if the new "proposed_strategy" passes 3 pre-commit external verifier checks (more on these verifier checks later), it will be promoted, and will become the new "benchmark_strategy", and then finally the git commit will happen. This will then allow us to go to the next round, where we will try an even newer "proposed_strategy" and check it's performance against the new "benchmark_strategy" using the same external evaluator checks, and promote it if the checks pass, and git commit again. Hence we would like to overhaul the current existing pre-commit check mechanism.



We will implement this plan in the following step:

Step 1: Understand the code, and precisely scope what it is that needs ablations to be iteratively improved.

Step 2: Setup 3 external verifier precommit checks. Each verifier will have 2 runs, ablating and comparing "proposed_strategy" vs "benchmark_strategy". Each of the 3 external-verifier pre-commit checks should use the exact same implementation for each ablation strategy.

Step 3: Implement the first iteration of the "proposed_strategy", with the aim to handle circular, curved and vertical lines better, while maintaining performance on horizontal lines.

Step 4: Setup strategy promotion mechanism and configuration (from "proposed_strategy" promotes to "benchmark_strategy") if pre-commit checks passes.

### STEP 1 
In this step, you should understand the code, and precisely scope what part of the pipeline is will be needing ablations to be iteratively improved:

We want to improve how this application segments text-lines (using the resized_images, heatmaps and the respective predictions of the GNN i.e the <Baseline/> in page-xml). In other words, we want to improve the part of the pipeline which converts the GNN predictions (<Baseline/> in page-xml) to the text-line images which are fed to the downstream OCR model for fine-tuning or inference.

Understand how the GNN predictions (<Baseline/> in page-xml)  are converted to text-line images, which are then fed to the downstream OCR model for fine-tuning or inference. We want to improve precisely this part of the pipeline.

Once done understanding, edit and refactor the code, such that it becomes modular and ablation friendly. Regarding this, it is very important to keep in mind that each of the 3 external-verifier pre-commit checks should use the _exact same implementation_ for each ablation strategy. So we want to setup the code in a way that when we will implement the "proposed_strategy", we will reuse the code of the "proposed_strategy" and "benchmark_strategy" for all 3 pre-commit checks.


### STEP 2 Implement 3 external verifier checks
I want you to overhaul the pre-commit evaluation, with one primary motive: Perform ablations on the text-line segmentation strategies - "benchmark_strategy" vs "proposed_strategy" new generalized text-line segmentation. In the future, when we want to try a newer version of text-line segmentation, the "proposed_strategy" will become the "benchmark_strategy", and the newer version will become the "proposed_strategy".

This will allow us to iteratively improve with each git commit in this branch. Hence we want to create overhaul the evaluation framework with the motto:
"every proposed will become a baselines, only to be replaced by a new proposed in the future."

The 3 external verifier checks which we want to implement are:
- pre-trained-pipeline gate (2 tests on 'eval_data', "benchmark_strategy" vs "proposed_strategy" ablation)
- active-learning-ocr-finetuning gate (2 tests on 'eval_data', "benchmark_strategy" vs "proposed_strategy" ablation)
- circular-layout-ocr-finetuning gate (2 tests on 'eval_data_2', "benchmark_strategy" vs "proposed_strategy" ablation)

We already have pre-trained-pipeline gate, and active-learning-ocr-finetuning gate implemented, and will need to modify them to fit this new evaluation framework. However, circular-layout-ocr-finetuning gate is not yet implemented, so we need to implement this from scratch.





### pre-trained-pipeline gate
For this gate, keep everything the same, other than the abltation.

### active-learning-ocr-finetuning gate
For this gate, keep everything the same, other than the abltation and one nuance: Note that for this test we consider that the GNN has done the upstream text-line detection (in gnn-format, <Baseline points="x1,y1 x2,y2 .."/> perfectly). For "benchmark_strategy" we currently directly crop out text-lines from the page-xml, as the page-xml of the eval-data has been prepared using the baseline method it self. For "proposed_strategy" ablation, we want to use the "<Baseline points="x1,y1 x2,y2 .."/> entries in the page-xml (which are essentially the GNN outputs). However, in the future, both "benchmark_strategy" and "proposed_strategy" methods will use <Baseline points="x1,y1 x2,y2 .."/>, the the current "proposed_strategy" method becomes the "benchmark_strategy" method

### circular-layout-ocr-finetuning gate
For this gate, keep everything the same, other than the abltation. For this too, we consider that the GNN has done the upstream text-line detection (in gnn-format, <Baseline points="x1,y1 x2,y2 .."/> perfectly).  For "benchmark_strategy" we currently directly crop out text-lines from the page-xml, as the page-xml of the eval-data has been prepared using the baseline method it self. For "proposed_strategy" ablation, we want to use the "<Baseline points="x1,y1 x2,y2 .."/> entries in the page-xml (which are essentially the GNN outputs). However, in the future, both "benchmark_strategy" and "proposed_strategy" methods will use <Baseline points="x1,y1 x2,y2 .."/>, the the current "proposed_strategy" method becomes the "benchmark_strategy" method

### What success means for each gate:
- pre-trained-pipeline gate - proposed as good as or better than baseline (slightly inferior also OK)
- active-learning-ocr-finetuning gate - proposed as good as or better than baseline (slightly inferior also OK)
- circular-layout-ocr-finetuning gate - proposed should be strictly better than baseline (slightly inferior NOT OK)

### OCR model config
'eval_dataset_v2' has only 5 pages. We want to use first 3 pages for iterative fine-tuning, and last 2 for verification and metric calculation (which will eventually help us select). Please use the same metrics and the best OCR model config we use in app/tests/test_recognition_finetuning_e2e.py.




### STEP 3 Implement Proposed_strategy
Every curved line, when seen locally is a straight line. We can get good information of the curvature using the 'Baseline' in the PAGE-XML (and the gnn-format labels). There are only two types of text-lines circle and curved line (of which straight line is a special case). Once we cut the circle, we want to treat it as a curved line. Hence use the ENGINEERING_DOCTRINE to find invariants, abstraction and tread every line the same (with minimal if/else edge case last mile handing).

Proposed Strategy:
- straight line is a special case of curved line
- a curved line is straight locally
- Once we cut the circle, we want to treat it as a curved line. Cut a circle always at the top-most point, as that's where we start annotating.
- keep the heuristic smartly crop out text coming from the "top" or "bottom" from adjacent lines. but generalize this for curved lines. This heuristic has been tuned for horizontal lines, having in mind that some scripts like devanagari have diacritic marks, matras extend out from the main text line (and which need to be included), but matras and diacritics from adjacent lines need to be excluded. Carefully study this.
- once cropped, we join all cropped images, then to convert this mask, to a rectangular image, we set it's background to the median color of the page.
- this generalized heuristic masking logic should be separate from the line straightening logic.
- When we will convert a curved line or a vertical line to a horizontal line as required, there is a chance that this text-line image will be upside down (as the conversion is ambiguous). line orientation domain knowledge:
    - reading order always left to right
    - always clockwise for circular layouts
    - make this configurable according to the script (right now it's right to left)
    - During training, use the CER of the orientations line to detect the right orientation. Then use these as labels (with OCR model final layer hidden states as inputs) to train a small MLP classifier to decide orientation during inference.

______
- Hence we will actually need to create new copies of the PAGE-XML (with updated Coords, everything else fixed) when trying different strategies. Keep the code standardized - hence one page-xml for "benchmark_strategy", and one for "proposed_strategy" - and then we straighten the line if is curved, circle or vertical. The benchmark_stratefy doesn't have any such handling of curved, circle or vertical.

(we do heurist adjacent line text cropping from top and bottom, but in a more generalized way to support curved line and vertical lines)
Step 1 is segmentation in page coordinate space. This step updates the `TextLine/Coords` polygons in copied PAGE-XML files. These `Coords` remain page-space polygons that describe where the text line is on the manuscript image.

(if line is circular, curverd or vertical, we do step 2)
Step 2 is unwrapping for OCR. This step consumes the page image plus the updated `Coords` and `Baseline`, then creates horizontal OCR-ready line images and metadata. The unwrapped horizontal rectangle must not be treated as the PAGE-XML `Coords`, because it no longer represents the original page-space layout.

Segmentation and OCR unwrapping are two separate steps.

Segmentation input:
- page image
- heatmap
- gnn points and text-line labels
- PAGE-XML `Baseline` and `TextEquiv`
- strategy config

Segmentation output:
- copied PAGE-XML with updated page-space `TextLine/Coords`
- page-space polygons only
- metadata including strategy name, line ids, cut point, tangent/normal settings, and source inputs

OCR unwrapping input:
- copied PAGE-XML
- page image
- page-space `Coords`
- `Baseline`
- unwrapping config

OCR unwrapping output:
- OCR-ready horizontal crop images
- `PreparedPageDataset`-compatible manifests
- orientation candidate metadata
- selected candidate and rejected candidate scores

The unwrapped horizontal rectangle must never be written as PAGE-XML `Coords`.

________________


The current segmentation behavior appears to work reasonably well on horizontal and vertical portions of circular text when judged in the global page frame. It crops too much or too little near diagonal tangent regions, especially around roughly 45, 135, 225, and 315 degrees. This supports the hypothesis that global x/y padding and rectangle-bridging are the wrong abstraction for curved text. New strategies should use local tangent and local normal directions along the baseline so padding means "along the line" and "across the line" rather than "page x" and "page y".




## Additional Notes:
Please clean up existing pre-commit checks and remove stale code. We only want the new 3 pre-commit checks mentions above going ahead.

Note that <Baseline points="x1,y1 x2,y2 .."/> are essentially processed gnn-format predictions, where all points, having the same label, are joined to form the polyline <Baseline points="x1,y1 x2,y2 .."/>


While doing this overhaul, please clean up any stale code if you find any. Follow ENGINEERING_DOCTRINE.md to keep the code base maintainable, and future proof. Do not be afraid to do a big overhaul if you think it will be more future proof, and more in spirit of this "benchmark_strategy" vs "proposed_strategy" ablation vision. MORE IMPORTANTLY, note that the current evaluation suite will only test the "benchmark_strategy" vs "proposed_strategy" in the context of text-lines segmentation (conversion of GNN output to OCR Model input). However, in the future "benchmark_strategy" vs "proposed_strategy" can be any ablation in the entire pipeline - you may also get more context regarding this from EVAL.md, and VISION.md. Do not let EVAL.md corrupt your context.

Refer to AGENTS.md to know common hiccups like which conda environment to use, how to fix for unicode output errors in windows, where to write.

Also please think critically and try to find flaws, bugs or unexpected subtle effects which might happen in upstream code or downstream code due to this implementation.






