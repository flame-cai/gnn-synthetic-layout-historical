# VISION.md

The Semi-Automatic Tool in 'app/' currently offers a way to digitize text from Historical Manuscript Images in two stages: 
1) Layout Analysis (text-line segmentation and text-region labeling, and creation of PAGE-XML)
2) Text-line Recogntion (recognizing text from text-lines segments and updating the PAGE-XML)

This semi-automatic app is thus is very "active learning" friendly for the following deep learning tasks:
1) character detection task (CRAFT)
2) The binary edge classification task (GNN)
3) The grouping of text-lines belonging to the same text-region task (this is done manually as of now, but the GNN can be trained to do this in a multi-task learning setup)
4) Text Recognition Task (EasyOCR or Gemini)

By Active Learning we mean that if the deep learning model makes mistakes, they are corrected by the human manually, which gives us more fine-tuning data, using which the deep learning models can learn to make less mistakes next-time, thus ultimately rapidly reducing the manual human effort required. 
Right now, the tool does not actually do active learning on any of the 4 tasks, but it does the ground work to enable it. The Vision is that if we are digitizing a manuscript with 10 pages, maximum manual effort will be required (accross all 4 tasks) for Page 1, then fine-tuning will happen, then less effort will be required for Page 2, the fine-tuning will happen again, then subsequent Pages will rapidly require drastically less manual effort. Hence we want to eventually enable rapid fine-tuning of CRAFT, the GNN (binary edge classification task, grouping text-lines into text-regions task), and the text-line recognition task. Note: CRAFT is not fine-tunable as the source code in not opensource, but we can implement a surrogate model which does what CRAFT does, and which can be iteratively trained in self supervised (using CRAFT outputs) or a supervised manned (human adding/deleting nodes)