This plan details specifications for how to implement fine-tuning capability for EasyOCR text-line recognition model (vadakautuhala.pth, which is names as Local OCR in the GUI), which recognizes the text from the segmented text-lines. This will drastically and interatively reduce the effort it takes for the human to make corrections to the recognized text from segmented text-lines (Think Active Learning).

To implement this, I want you first implement the fine-tuning code and evaluation test without using GUI at all, using evaluation dataset: 'app\tests\eval_dataset' (although we want to eventually allow the user to optionally fine-tune the recognition model during each page save, such the predictions on subsequent pages will be more accurate in an active learning sense). 
'app\tests\eval_dataset\images' contains the original images, and 'app\tests\eval_dataset\labels\PAGE-XML' contain the text-line bounding polygons (Coords) and the ground-truth text (TextEquiv) in  (Unicode). We want to use these bouding polygons to crop out text-line images (in the same the 'app' already does), and use the Unicode ground-truth text as the labels, for each text-line, in each page. 

Hence the text-line image would be the Input
Ground-truth text in Unicode would be the Label

You will have to write conversion script to robust prepare the data in a way the EasyOCR model vadakautuhala.pth requires it in.
Please find reference source code for training/fine-tuning vadakautuhala.pth here (you may clone the repo):
https://github.com/flame-cai/case-study-handwritten-sanskrit-ocr/tree/main


# Verification
To verify if you have correctly implemented the fine-tuning EasyOCR feature, please use a more advanced version of the evaluator test:
app\tests\test_ci_e2e.py

## What is different:
1) The test app\tests\test_ci_e2e.py use CRAFT+GNN to segment text-lines (which can also also make mistakes). In this test, we don't want this - we want the text-line segmentation to be perfect (used from the ground-truth PAGE-XML, but processed in a similar way the tool does. Hence please carefully implement the text-line image extraction code.).

2) The test app\tests\test_ci_e2e.py evaluates the performace of the entire pipeline on all 15 pages at once using only pre-trained models (CRAFT, v2.pt, vadakautuhala.pth). There is no fine-tuning happening. No active learning happening. Hence this is going to be the most important difference:

The new test should do the following:

We have 15 total pages in eval_dataset.


STEP 0:

0) Use pre-trained vadakautuhala.pth to recognize text from text-line images from PAGES 10-15. Evaluate against ground truth, and log Page-Level CER, time-taken, and other metrics refering to evaluate.py.


STEP 1
1a) Fine-tune vadakautuhala.pth on all text-line pairs from PAGE 1.
1b) Use the 1 PAGE Fine-tuned vadakautuhala.pth to recognize text from text-line images from PAGES 10-15. Evaluate against ground truth, and log Page-Level CER, time-taken, and other metrics refering to evaluate.py.
1c) Compare, Check and Verify if Fine-tuning on PAGE 1 helped. Log the results.

Proceed to next step if performance 1 PAGE Fine-tuned vadakautuhala.pth is better than the pre-trained vadakautuhala.pth (on PAGES 10-15)

STEP 2
2a) Fine-tune vadakautuhala.pth on all text-line pairs from PAGE 1 and 2.
2b) Use the 2 PAGE Fine-tuned vadakautuhala.pth to recognize text from text-line images from PAGES 10-15. Evaluate against ground truth, and log Page-Level CER, time-taken, and other metrics refering to evaluate.py.
2c) Compare, Check and Verify if Fine-tuning on 2 PAGES helped.

Proceed to next step if performance 2 PAGE Fine-tuned vadakautuhala.pth is better than 1 PAGE Fine-tuned vadakautuhala.pth (on PAGES 10-15)

STEP 3
3a) Fine-tune vadakautuhala.pth on all text-line pairs from PAGE 1 and 2 and 3.
3b) Use the 3 PAGE Fine-tuned vadakautuhala.pth to recognize text from text-line images from PAGES 10-15. Evaluate against ground truth, and log Page-Level CER, time-taken, and other metrics refering to evaluate.py.
3c) Compare, Check and Verify if Fine-tuning on 3 PAGES helped.

Proceed to next step if performance 3 PAGE Fine-tuned vadakautuhala.pth is better than 2 PAGE Fine-tuned vadakautuhala.pth (on PAGES 10-15)

So on...till we fine-tune of 5 PAGES.

Please version all fine-tuned models, and log experiment results like an expert researcher. Do not overwrite the original vadakautuhala.pth. Also plot a neat figure, which shows the performance expected drop in Page-level CER with increasing fine-tuning data. Make sure the experiment is well thought out, robust, and replicable. 

IMPORTANT: Failure to pass this evaluation experiment is your signal that the fine-tuning logic has not been implemented correctly, and that you need to investigate the cause, and try again. Keep on iterating till this experiment passes successfully.

Hyperparameter Note:
Please use batch size of 1 for both training and inference of the recogntion model.








