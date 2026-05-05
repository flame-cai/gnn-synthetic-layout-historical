



We want to take inspiration from `app/tests/test_recognition_finetuning_e2e.py`, in the sense that we would like to experiment with various parts of the recognition and active pipeline and select the best configure using the external verification metrics (GUI-free headless manner). However instead of working with 'app\tests\eval_dataset', will be working with 'app\tests\eval_dataset_v2', which is a different file structure, and needs to be handled differently as mentioned below:

1) In 'test_recognition_finetuning_e2e.py' we assume the text line segmentation to be perfect, and directly crop out the lines from the page using the PAGE-XML (the same way the GUI tool does it.) However, as eval_dataset_v2 contains circular text-lines, vertical text-lines, and curvy text-lines - we can assume the labels in the gnn-format (eval_dataset_v2\layout_analysis_output\gnn-format) to be the perfect ground truth. However, the way we segment text-lines using these ground truth labels in gnn-format, and the heatmaps (eval_dataset_v2\heatmaps) works well only for relatively straight, horizontal lines. It fails for text-lines in a circular layout, very curved lines, and vertical lines. Hence _this conversion pipeline_ from gnn-format labels, to the text-line images (similar to the ones we use in test_recognition_finetuning_e2e.py) needs to optimized and evolutionarily searched - because we have the downstream ground-truth text content of each page in the PAGE-XML files in (eval_dataset_v2\layout_analysis_output\page-xml-format).

Note that the PAGE-XML files contain the grount-truth text (TextEquiv), and also the ground-truth polylines (Baseline) based on the gnn-format labels. However the (Coords) are not the groun-truth bounding polygons. These Coords denote the bounding polygons obtained using the current flawed pipeline, which only work well for relatively straight, horizontal lines, and not circular, curved, or vertical lines. 

PcGts
├── @xmlns
├── Metadata
│   ├── Creator
│   └── Created
└── Page
    ├── @imageFilename
    ├── @imageWidth
    ├── @imageHeight
    └── TextRegion*
        ├── @id
        ├── @custom
        ├── Coords
        │   └── @points
        └── TextLine*
            ├── @id
            ├── @custom
            ├── Baseline
            │   └── @points
            ├── Coords
            │   └── @points
            └── TextEquiv
                └── Unicode

For reference gnn-format labels are as follows:
- page_2_dims.txt (heatmap image dimension)
1250.0 1250.0

- page_2_inputs_normalized.txt (coordinate x, coordinate y, font_size)
0.489600 0.077600 0.000004
0.452000 0.078400 0.000004
0.470400 0.078400 0.000005
...


- page_2_inputs_unnormalized.txt (coordinate x, coordinate y, font_size)
612.000000 97.000000 0.005600
565.000000 98.000000 0.005600
588.000000 98.000000 0.006400
...

- page_2_labels_textline.txt (points belonging to the same text-line have the same label)
0
0
0
...


IMPORTANT:
The existing heuristic text-line segmentation logic works well for horizontal lines. It also effectively uses the heatmap, along with additional heuristics to not include text from adjacent (above and below) lines. Please study this, and attempt to generalize this strategy to any curved text-line (including circular text lines), WITHOUT making it worse on horizontal lines.
- Treat curved text-line as a generalization. That is, a straight line is a special case of the curved line.
- For a circular text line, we will have to 'cut' it at the topmost point, to convert it into a curved line topologically.
- We must make sure this generalization does not affect the performance on horizontal lines in 'eval_dataset'. So the previous tests should pass after this change (eventually)
- When we will convert a curved line or a vertical line to a horizontal line as required, there is a chance that this text-line image will be upside down (as the conversion is ambiguous). To fix this, we can use the _uncertainty_ of the OCR model to decide which orientation is the right one.
- Hence we will actually need to create new copies of the PAGE-XML (with updated Coords, everything else fixed) when trying different strategies. 


INSPIRATION:
Every curved line, when seen locally is a straight line. We can get good information of the curvature using the 'Baseline' in the PAGE-XML (and the gnn-format labels). There are only two types of text-lines circle and curved line (of which straight line is a special case). Once we cut the circle, we want to treat it as a curved line. Hence use the ENGINEERING_DOCTRINE to find invariants, abstraction and tread every line the same (with minimal if/else edge case last mile handing). Perhaps we can also take inspiration from the metric-tensor.



2) Hence, in this experiment, we do not want to optimize the OCR model fine-tuning configuration. Instead we want to optimize :
- how vertical, curved, and circular text-line images are prepared from the gnn-format ground-truths.
- how vertical, curved, and circular text-line images are converted to horizontal text-line image rectangles (as required to finetune and run inference by the OCR model)
- the converted horizontal line orientation ambiguity decision using OCR model _uncertainty_. So we will need to try out both orientations.


EVALUATION, VERIFICATION, EVOLUTIONARY SEARCH.
Every strategy we try, should do well on the following tests:
- pre-trained models test (with attempted new generalized line-segmentation using gnn predictions and heatmap)
- active learning OCR model test on 'eval_dataset_v2' (using ground truth gnn-format labels + attempted new generalized line-segmentation using gnn predictions and heatmap. New copies of the PAGE-XML ground truth will be created.)
- Do not check using the the active learning OCR model test on 'eval_dataset'. As it is time consuming, and can be done later.

'eval_dataset_v2' has only 5 pages. We want to use first 3 pages for iterative fine-tuning, and last 2 for verification and metric calculation (which will eventually help us select). Please use the same metrics and the best OCR model config we use in app/tests/test_recognition_finetuning_e2e.py.

This is an experiment which combines the generative capabilities of LLMs with external verifier metrics to perform evolutionary search in python code space. Please prepare this experiment as an expert researcher with good logging, configuration and segmentation strategy handling. Please create a new directory 'circular_OCR_test' for this experiment. The directory will have a file IDEAS.md which both you and the human can read/write/update with new ideas we have to try out. This IDEAS.md will be like a good genetic 'pool'. Only the good ideas survive, the bad ones will be removed. This will allow us to juxtapose ideas and iterate. Please think carefully and prepare this evolutionary experiment system well.
Only make modifications inside 'app/', but feel free to change any code inside it for each experiment. 

IMPORTANT WORKFLOW INSTRUCTIONS: 
Initially, do not perform multiple experiments fully autonomously. At the start I would like the following workflow:
- we want to have this as a new pre-commit check, along with the existing pre-trained model only check. 'test_recognition_finetuning_e2e.py' will be disabled as it is time-consuming.
- you will then change everything that needs to be optimized, and then I will manually commit. For the commit to pass, performance on the new pre-commit check must have improved than before, and performance on the existing pre-trained check should not worsen.
- once the commit passes. I will then manually update the IDEAS.md file in 'circular_OCR_test', and then I will ask you to again make optimizations based on it. Then I will manually commit again with hope that it passes.
- after a few iterations, i might ask you to go full automatic mode and try out multiple experiments. But we don't want this at this start. 
- please write any other scaffolding documents with instructions other than IDEAS.md which will help with this experiment. This will help not dilute the context, and allow reuse.



We must make sure that each strategy we try does not break any upstream or downstream code.


