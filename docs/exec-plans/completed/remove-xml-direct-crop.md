I want you to make modify the Recognition Fine-Tune Gate, which uses the dataset: app\tests\eval_dataset

Right now for this check we assume that the text-line segmentation is perfect, and hence we directly use the ground truth bounding polygons <Coords points=../> from the page-xml files as a starting point to process text-line images to be fed to the OCR model for fine-tuning. We want to change this.
Instead of using the <Coords points=../> I want you to use the ground truth <Baseline points="x1,y1 x2,y2 .."/> as the starting point. We want to process the <Baseline points="x1,y1 x2,y2 .."/> in the page-xml, the heatmap, the original image using a pipeline to actually get the <Coords points=../>. This pipeline should be exactly the same as the pipeline in the application, and the pre-trained only pre-trained only check. Hence we should ideally get the _exact same_ <Coords points=../>, and the eventual processed text-line images for the OCR model after this modification.

So instead of taking the shortcut of directly using the <Coords points=../>, I want the test to take the longcut, and calculate the <Coords points=../> from the <Baseline points="x1,y1 x2,y2 .."/>, the heatmap, the original image, using the pipeline which the application actually uses. We must ensure that the longcut get us the exact same <Coords points=../> which the shortcut used directly.

Hence, ideally, we want to use this condition for getting equivalent outputs (<Coords points=../>, processed text-line images for OCR model) to verify if this update has passed.

Once passed, please clean up stale code, and update the docs.

So after this update, both pre-commit checks will ideally be using equivalent pipelines to get the text-line images, which are to be fed to the OCR model.

IMPORTANT.
To ensure equivalency, you will also need to study how the application pipelines works, especially when the user manually adds or deletes nodes, and how these are processed heuristically. Because sometimes, the heatmaps makes mistakes, so the user has to manually adds and deletes nodes, to get the ground-truth text-line <Baseline points="x1,y1 x2,y2 .."/>. The <Baseline points="x1,y1 x2,y2 .."/>s in the page-xml are already ground-truth, but we don't know which nodes, from which baseline were manually added by the user, and which were deleted. So we need to handle this somehow. _Perhaps, we can use a slightly looser notion of equivalency to account for this_. Please think hard about this before implementation. In any case, we want to start from <Baseline points="x1,y1 x2,y2 .."/> which are the ground-truths, and we want to prioritise this over perfect equivalency.

In this regard, it is also important to note that <Baseline points="x1,y1 x2,y2 .."/> are essentially processed the gnn-format predictions, where all points, having the same label, are joined to form the polyline <Baseline points="x1,y1 x2,y2 .."/>, which denotes a text-line location.

While making this change, please ask me any clarifications if you feel the need to.

- use conda environment gnn_layout
- avoid windows permission denied error, write everything to the current directory only.no saving in C:\\temp
- the "images" in eval_dataset is the same as the "images_resized" in input_manuscrips folder used in the production application
