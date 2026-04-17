# Plan Introduction

This plan details specifications for how to implement a fine-tuning capability for the text-line recognition model (vadakautuhala.pth, which is names as Local OCR in the GUI), which recognizes the text from the segmented text-lines. This will interatively reduce the effort it takes for the human to make corrections to the recognized text from segmented text-lines (Think Active Learning).

This plan is concerned with improving the ## 🧩 **Semi Automatic Annotation Tool ```app/```** as mentioned in AGENTS.md file, with the ability to fine tune the text-line recognition model, which recognizes text from the segmented text-lines from the previous step - to enable active learning.

To implement this, I want you to implement the fine-tuning code, and also implement an evaluation experiment test to verify successful implementation of the fine-tuning code using the evaluation dataset: 'app\tests\eval_dataset' (For the finetuning, or the verifier evaluation experiment, we won't be using the GUI at all - although we want to eventually allow the user to optionally fine-tune the recognition model (Local OCR) during each page save, such the predictions on subsequent pages will be more accurate in an active learning sense). 

in eval_dataset:
'app\tests\eval_dataset\images' contains the original images, and 'app\tests\eval_dataset\labels\PAGE-XML' contain the text-line bounding polygons (Coords) and the ground-truth text (TextEquiv) in  (Unicode). We want to use these bouding polygons to crop out text-line images (in the same the 'app' already does), and use the Unicode ground-truth text as the labels, for each text-line, in each page. Please carefully parse the PAGE-XML files and input images, and convert them to the required format. Please carefully implement the text-line image extraction code. In other words, we want the text-line images extracted using the ground truth PAGE-XML to look and be processed exactly like the text-line images processed by the tool which it saves in 'input_manuscripts\testing\layout_analysis_output\image-format\1481_0011\textbox_label_0\line_1.jpg' for example.

PAGE-XML format reference:
<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">

  <Metadata>
    <Creator>YOUR_TOOL</Creator>
    <Created>YYYY-MM-DDTHH:MM:SS</Created>
  </Metadata>

  <Page imageFilename="image.jpg" imageWidth="WIDTH" imageHeight="HEIGHT">

    <TextRegion id="region_1" custom="label">
      <Coords points="x1,y1 x2,y2 x3,y3 x4,y4"/>

      <TextLine id="line_1">
        <TextEquiv>
          <Unicode>TEXT_HERE</Unicode>
        </TextEquiv>
        <Baseline points="x1,y1 x2,y2"/>
        <Coords points="..."/>
      </TextLine>

    </TextRegion>

  </Page>
</PcGts>


You will have to write conversion script to robust prepare the data in a way the finetuning code for vadakautuhala.pth requires it in:

The structure of the expected data folder as below:
data
├── gt.txt
└── test
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
    └── ...

At this time, gt.txt should be {imagepath}\t{label}\n
For example:
test/word_1.png Tiredness
test/word_2.png kills
test/word_3.png A
...

**Important**: To get the data in this structure, you will have to order the text-lines of each image, using the respective PAGE-XML file, using the same logic 'app\tests\evaluate.py' uses to calculate the Page-level CER.

---

**Important**: Please find the actual reference source code for training/fine-tuning text-line recognition models like vadakautuhala.pth here:
C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\recognition_finetuning_ref\finetuning_reference_1
C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\recognition_finetuning_ref\finetuning_reference_2

**Recognition model Notes:**
Please use **batch size of 1 for both training and inference of the recogntion model**.
Make changes in recognition source code only where required, try to reuse as much files as possible. 

**Important**: This has been modified by our current repo to specialize in recognizing Sanskrit Characters from Sanskrit text-line images: notice the  --character argument in python deep-text-recognition-benchmark/train.py in train.sh. Keep it this way for now. But write code in such that in the future we can configure it to work with a different script. Hence please be carefull and only make changes where required.
Reuse existing files from the current repo as much as possible:
C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\app\recognition
Reuse the --character arguement by copying it very carefully! Do not recall from memory, COPY IT.



# Evaluation Experiment
To verify if you have correctly implemented the fine-tuning of the text-line recognition feature, please design and implement more advanced version of the evaluator test:
app\tests\test_ci_e2e.py

## What is different:
1) The test app\tests\test_ci_e2e.py use CRAFT+GNN to segment text-lines (which can also also make mistakes). In this test, we don't want this - we want the text-line segmentation to be perfect (used from the ground-truth PAGE-XML, but processed in a similar way the tool does. Hence please carefully implement the text-line image extraction code. In other words, we want the text-line images extracted using the ground truth PAGE-XML to look and be processed exactly like the text-line images the tool saves in 'input_manuscripts\testing\layout_analysis_output\image-format\1481_0011\textbox_label_0\line_1.jpg' for example.)

2) The test app\tests\test_ci_e2e.py evaluates the performace of the entire pipeline on all 15 pages at once using only pre-trained models (CRAFT, v2.pt, vadakautuhala.pth). There is no fine-tuning happening. No active learning happening. Hence this is going to be the most important difference:

The new test should do the following:

We have 15 total pages in eval_dataset.

STEP 0:

0) Load for inference, and use the pre-trained vadakautuhala.pth to recognize text from text-line images from PAGES 10-15. Evaluate against ground truth, and log Page-Level CER, time-taken, and other metrics refering to evaluate.py.


STEP 1
1a) Fine-tune vadakautuhala.pth on all text-line pairs from PAGE 1.
1b) Load for inference, and use the 1 PAGE Fine-tuned vadakautuhala.pth to recognize text from text-line images from PAGES 10-15. Evaluate against ground truth, and log Page-Level CER, time-taken, and other metrics refering to evaluate.py.
1c) Compare, Check and Verify if Fine-tuning on PAGE 1 helped. Log the results.

Proceed to next step if performance 1 PAGE Fine-tuned vadakautuhala.pth is better than the pre-trained vadakautuhala.pth (on PAGES 10-15)

STEP 2
2a) Fine-tune the 1 PAGE Fine-tuned vadakautuhala.pth on all text-line pairs from PAGE 2.
2b) Load for inference, and use the 2 PAGE Fine-tuned vadakautuhala.pth to recognize text from text-line images from PAGES 10-15. Evaluate against ground truth, and log Page-Level CER, time-taken, and other metrics refering to evaluate.py.
2c) Compare, Check and Verify if Fine-tuning on 2 PAGES helped.

Proceed to next step if performance 2 PAGE Fine-tuned vadakautuhala.pth is better than 1 PAGE Fine-tuned vadakautuhala.pth (on PAGES 10-15)

STEP 3
3a) Fine-tune the 2 PAGE Fine-tuned vadakautuhala.pth on all text-line pairs from PAGE 3.
3b) Load for inference, and use the 3 PAGE Fine-tuned vadakautuhala.pth to recognize text from text-line images from PAGES 10-15. Evaluate against ground truth, and log Page-Level CER, time-taken, and other metrics refering to evaluate.py.
3c) Compare, Check and Verify if Fine-tuning on 3 PAGES helped.

Proceed to next step if performance 3 PAGE Fine-tuned vadakautuhala.pth is better than 2 PAGE Fine-tuned vadakautuhala.pth (on PAGES 10-15)

So on...till we iteratively fine-tune till 5 PAGES.

Please version all fine-tuned models, and log experiment results in detail like an expert researcher. Do not overwrite the original vadakautuhala.pth. Also plot a neat figure, which shows the expected drop in Page-level CER with increasing fine-tuning data. Make sure the experiment is well thought out, robust, and replicable. 

**IMPORTANT**: Failure to pass this evaluation experiment is your signal that the fine-tuning logic has not been implemented correctly, and that you need to investigate the cause, and try again. Keep on iterating till this experiment passes successfully.

**IMPORTANT**: This evaluation test current works with one dataset 'eval_dataset', but we want to write the test such that it can be configured to work with multiple dataset which we will add later.

Please feel free to study the current codebase and open relevant files to open it:
- C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\app\recognition
- C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\app\app.py
- and anything else you feel is relevant

In the broader specifications mentioned in EVAL.md This specific evaluation is only for fine-tuning the recognition model, with all other models being pre-trained (CRAFT, v2).


# Meta Specifications:
Hence you will do both things, implement the fine-tuning functionality, and also implement the evaluation experiment which will give you the signal whether you have implemented the fine-tuning capability correctly. Important: Although everything we do here has no GUI, we want eventually want the user optionally turn on an Active Learning mode, where saving a page, will start the fine-tuning of the recognition Local OCR model _in the background_, and once done, it will use it to better recognize text from text-lines from subsequent pages.

Please keep track of what changes you implemented after each evaluation experiment failure in a markdown file.

please use the conda environment 'gnn_layout' in powershell to run scripts and commands. Feel free to install dependencies if required.












