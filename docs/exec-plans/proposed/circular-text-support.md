This research plan will enable us to combine the generative capabilities of LLMs with external verifier metrics to perform step by step evolutionary search in python code space - with the purpose to improve how this application segments text-lines (using the resized_images, heatmaps and the respective predictions of the GNN i.e the <Baseline points="x1,y1 x2,y2 .."/> in page-xml). In other words, we want to improve the part of the pipeline which converts the GNN predictions <Baseline points="x1,y1 x2,y2 .."/> in page-xml) to the text-line images which are fed to the downstream OCR model for fine-tuning or inference.

The application already has such a text-line segmentation strategy. We will call this the "benchmark_strategy". However this strategy is flawed and works only for horizontal text-lines (not circular, curved, or vertical lines). To fix these flaws, we will be implementing a new "proposed_strategy". The goal is that if the new "proposed_strategy" passes 3 pre-commit external verifier checks (more on these verifier checks later), it will be promoted, and will become the new "benchmark_strategy", and then finally the git commit will happen. This will then allow us to go to the next round, where we will try an even newer "proposed_strategy" and check it's performance against the new "benchmark_strategy" using the same external evaluator checks, and promote it if the checks pass, and git commit again. Hence we would like to overhaul the current existing pre-commit check mechanism.



We will implement this plan in the following step:

Step 1: Understand the code, and precisely scope what it is that needs ablations to be iteratively improved.

Step 2: Setup 3 external verifier precommit checks. Each verifier will have 2 runs, ablating and comparing "proposed_strategy" vs "benchmark_strategy". Each of the 3 external-verifier pre-commit checks should use the exact same implementation for each ablation strategy.

Step 3: Implement the first iteration of the "proposed_strategy", with the aim to handle circular, curved and vertical lines better, while maintaining performance on horizontal lines.

Step 4: Setup strategy promotion mechanism and configuration (from "proposed_strategy" promotes to "benchmark_strategy") if pre-commit checks passes.

### STEP 1 
In this step, you should understand the code, and precisely scope what part of the pipeline is will be needing ablations to be iteratively improved:

We want to improve how this application segments text-lines (using the resized_images, heatmaps and the respective predictions of the GNN i.e the <Baseline points="x1,y1 x2,y2 .."/> in page-xml). In other words, we want to improve the part of the pipeline which converts the GNN predictions (<Baseline/> in page-xml) to the text-line images which are fed to the downstream OCR model for fine-tuning or inference.

Understand how the GNN predictions (<Baseline points="x1,y1 x2,y2 .."/> in page-xml)  are converted to text-line images, which are then fed to the downstream OCR model for fine-tuning or inference. We want to improve precisely this part of the pipeline.

Once done understanding, edit and refactor the code, such that it becomes modular and ablation friendly. Regarding this, it is very important to keep in mind that each of the 3 external-verifier pre-commit checks should use the _exact same implementation_ for each ablation strategy. So we want to setup the code in a way that when we will implement the "proposed_strategy", we will reuse the code of the "proposed_strategy" and "benchmark_strategy" for all 3 pre-commit checks.


### STEP 2 Implement 3 external verifier checks
I want you to overhaul the pre-commit evaluation, with one primary motive: Perform ablations on the text-line segmentation strategies - "benchmark_strategy" vs "proposed_strategy" new generalized text-line segmentation. In the future, when we want to try a newer version of text-line segmentation, the "proposed_strategy" will become the "benchmark_strategy", and the newer version will become the "proposed_strategy".

This will allow us to iteratively improve with each git commit in this branch. Hence we want to create overhaul the evaluation framework with the motto:
"every proposed will become a baselines, only to be replaced by a new proposed in the future."

The 3 external verifier checks which we want to implement are:
- pre-trained-pipeline gate (2 tests on 'eval_data', "benchmark_strategy" vs "proposed_strategy" ablation)
- active-learning-ocr-finetuning gate (2 tests on 'eval_data', "benchmark_strategy" vs "proposed_strategy" ablation)
- circular-layout-ocr-finetuning gate (2 tests on 'eval_data_2', "benchmark_strategy" vs "proposed_strategy" ablation)

We already have pre-trained-pipeline gate, and active-learning-ocr-finetuning gate implemented, and will need to modify them to fit this new evaluation framework. However, circular-layout-ocr-finetuning gate is not yet implemented, so we need to implement this from scratch:

#### implementation details:

##### pre-trained-pipeline gate
This gate works with tests\eval_dataset.
Modify this gate such that we scope out which part of the pipeline we are improving upon iteratively and make it modular such that doing ablations is easy.

##### active-learning-ocr-finetuning gate
This gate works with tests\eval_dataset.
Modify this gate such that we scope out which part of the pipeline we are improving upon iteratively and make it modular such that doing ablations is easy.

##### circular-layout-ocr-finetuning gate
This gate works with tests\eval_dataset_v2. 'eval_dataset_v2' has only 5 pages. We want to use first 3 pages for iterative fine-tuning, and last 2 for verification and metric calculation (which will be used to compare "benchmark_strategy" with "proposed_strategy"). Other than this difference, this test is very similar to active-learning-ocr-finetuning gate, so please use the same metrics and the same OCR model fine-tuning config and recipe.

Write this gate such that we scope out which part of the pipeline we are improving upon iteratively and make it modular such that doing ablations is easy.


#### What success means for each gate:
- pre-trained-pipeline gate - "proposed_strategy" as good as or better than "benchmark_strategy" (slightly inferior also OK)
- active-learning-ocr-finetuning gate - "proposed_strategy" as good as or better than "benchmark_strategy" (slightly inferior also OK)
- circular-layout-ocr-finetuning gate - "proposed_strategy" should be strictly better than "benchmark_strategy" (slightly inferior NOT OK)


### STEP 3 Implement Proposed_strategy
We want to implement a proposed strategy which will take in the <Baseline points="x1,y1 x2,y2 .."/> from the page-xml, the heatmap, the original image, and then output text-line images in a format which can be fed to the downstream OCR model for finetuning or inference.

The proposed strategy should be able to handle vertical text, curved text, and circular text, along with standard horizontal text. For this our main inspiration is:
- Every curved line, when seen locally is a straight line. We can get good information of the curvature using the 'Baseline' in the PAGE-XML
- straight line is a special case of curved line

Hence we want this proposed_strategy to be a generalization of the current benchmark_strategy. In other words, the benchmark_strategy should be a special case of the proposed_strategy.

##### heuristic-adjacent line-text-cropping:
We will want to use a generalized version of the heuristic-adjacent line-text-cropping which the benchmark_strategy uses:

This heuristic smartly crops out text coming from the "top" or "bottom" from adjacent lines. But generalize this for curved lines, as this heuristic has been tuned for horizontal lines, having in mind that some scripts like devanagari have diacritic marks, matras extend out from the main text line (and which need to be included), but matras and diacritics from adjacent lines need to be excluded (they come from the "top and bottom"). Carefully study this.

##### OCR unwrapping
For horizontal line, we assume no OCR unwrapping would be required. For vertical line OCR unwrapping would just be rotation. For curved lines we will need an OCR unwrapping strategy.



#### How to handle each type of lines:
Hence use the ENGINEERING_DOCTRINE to find invariants, abstraction and treat every line the same (with minimal if/else edge case last mile handing).

##### Horizontal lines: 
We will use the same heuristic-adjacent-line-text-cropping. These lines should be handled exactly like how the benchmark_strategy handles them. 

##### Vertical lines: 
heuristic-adjacent-line-text-cropping needs to be modified such that it works with Vertical lines. Then these need to be converted to horizontal lines. The right orientation after conversion is ambiguous. We need orientation handling mechanism.

##### Curved lines:
STEP 1: heuristic-adjacent-line-text-cropping 
This needs to be modified such that it works with curved lines (and lines of shape S and C and U etc). 
The current version of the heuristic-adjacent-line-text-cropping crops too much or too little near diagonal tangent regions, especially around roughly 45, 135, 225, and 315 degrees. This supports the hypothesis that global x/y padding and rectangle-bridging are the wrong abstraction for curved text. New strategies should use local tangent and local normal directions along the baseline so padding means "along the line" and "across the line" rather than "page x" and "page y". Once heuristic-adjacent-line-text-cropping is done (we will get the <Coords/> in the page-xml and we can create new page-xmls with everything else the same but with new <Coords/>). These `Coords` remain page-space polygons that describe where the text line is on the manuscript image.

STEP 2: OCR unwrapping
Step 2 is unwrapping for OCR. This step consumes the page image, the updated `Coords`, and the `Baseline` and then then creates horizontal OCR-ready line images and metadata. The horizontal OCR-ready line images should contain ONLY the contents of the respective line's new `Coords` in an unwrapped transformed way. Once the new `Coords` are transformed and unwrapped, to we will fit a tight enclosing rectangle around them.
The background of this rectangular horizontal OCR-ready line images (the part outside the transformed unwrapped `Coords`) should be set to the median background color of the page the text-line is from.

Hence the workflow should be something like:

heuristic-adjacent-line-text-cropping and OCR unwrapping are two separate steps.

heuristic-adjacent-line-text-cropping input:
- page image
- heatmap
- PAGE-XML `Baseline` and `TextEquiv`
- strategy config

heuristic-adjacent-line-text-cropping output:
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

The unwrapped horizontal rectangle must never be written as PAGE-XML `Coords`.

The right orientation after conversion of curved lines too is ambiguous. We need orientation handling mechanism.


##### Circular lines:
We will cut a circle always at the top-most point, as that's where we start annotating the text.
Once we cut the circle, we want to process it just like a curved line. 


#### orientation handling mechanism
After the two steps (heuristic-adjacent-line-text-cropping and OCR unwrapping), the step of orientation handling still remains, as there is a chance that this text-line image will be upside down (as the conversion is ambiguous).

There are 4 possible orientation ambiguities, but we can narrow them down using domain knowledge which can be configured. Right now we would be processing Sanskrit text lines. So:
- reading order will always left to right
- always clockwise for circular layouts
This info, which will be configurable will be used to reduce the ambiguity. 

But some ambiguity will still remain. To fix this perhaps we can run the OCR model on all possible orientations, and then "smartly" select the right one based on the statistics OCR model outputs, and the OCR model uncertainty. To get these statistics, we can also perhaps use the three page training data in "eval_dataset_v2"


### Step 4:
Setup strategy promotion mechanism and configuration (from "proposed_strategy" promotes to "benchmark_strategy") if pre-commit checks passes. Perform clean up and additional checks. Once everything is done, prompt use to manually commit. This should trigger the pre-commit checks, and if they pass, the strategy promotion will happen. If they fail, we should be able to see good logs on what failed. Write good configuration files for the strategy promption framework.



## Additional Notes:
Note that <Baseline points="x1,y1 x2,y2 .."/> are essentially processed gnn-format predictions, where all points, having the same label, are joined to form the polyline <Baseline points="x1,y1 x2,y2 .."/>

Please clean up existing pre-commit checks and remove stale code. We only want the new 3 pre-commit checks mentions above going ahead.

While doing this overhaul, please clean up any stale code if you find any. Follow ENGINEERING_DOCTRINE.md to keep the code base maintainable, and future proof. Do not be afraid to do a big overhaul if you think it will be more future proof, and more in spirit of this "benchmark_strategy" vs "proposed_strategy" ablation vision. MORE IMPORTANTLY, note that the current evaluation suite will only test the "benchmark_strategy" vs "proposed_strategy" in the context of text-lines segmentation (conversion of GNN output to OCR Model input). However, in the future "benchmark_strategy" vs "proposed_strategy" can be any ablation in the entire pipeline - you may also get more context regarding this from EVAL.md, and VISION.md. Do not let EVAL.md corrupt your context.

Refer to AGENTS.md to know common hiccups like which conda environment to use, how to fix for unicode output errors in windows, where to write.

Also please think critically and try to find flaws, bugs or unexpected subtle effects which might happen in upstream code or downstream code due to this implementation. Please ask me any clarifications if you feel the need to.

use conda environment gnn_layout

avoid windows permission denied error, write everything to the current directory only.no saving in C:\\temp



