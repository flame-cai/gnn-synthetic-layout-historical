I want your help in updating the code for the experiment which compares the downstream performance of end-to-end VLMs with various variations of a traditional Annotation tool. 
The annotation tool uses CRAFT + GNN to perform layout analysis by performing binary edge classification, which decided which nodes are part of the same text-line. This essentially becomes the PAGE-XML Baseline. Once the text-lines are processed to form text-line image, we use an OCR model to predict the unicode text content from the text-line images.

Right now, in the Annotation tool experiment code, we are finetuning the OCR model on 0,1,2,3 pages. The change we want to do now, is that we also want to fine-tune the GNN on 1,2,3 pages too! (by augmenting each page 50 times)

Hence, we now want to fine-tune both, the OCR model, and also the GNN for binary edge classification.

Please keep the following in mind while implementing:
- keep original GNN model unchanged
- load model properly
- augment page 50 times before finetuning, also some percentage of pages from previous pages if any.
- carefully use the GNN configuration file, and the augmentation configuration file for fine-tuning/
- we should be able to use "\layout_analysis_output\gnn-format" labels to fine-tune the GNN model, for the respective fold, on the respective number of pages.


Please study the code, understand the task, and ask me for clarifications if required.

Once implemented, we want to first test of 1 fold, for all three manuscripts and check if it's working. We will check this by looking at the column (no-layout-postcorrection) for rows pages fine-tuned 0,1,2,3. Name this experiment as "GNN_finetune_1_fold_test". Once this is done, and I want to verify the results myself. Then we might adjust the fine-tuning hyperparamenter for the GNN iteratively. Finally, we will continue to run a full 5 fold experiment with all competing methods and ablations including VLM.

Note that this change should not we another ablation of the Annotation tool. This change should update the Annotation tool, finetuning, no-layout-postcorrection abltation itself. It's just that we are fine-tuning both, the GNN and the OCR model, instead of just the OCR model.

Follow experiment hygiene ( data split folds etc). This change, although big, should not essentially change the experiment setup. We are just changing what it means for a page to be fine-tuned. So how we are doing data splits should remain unchanged.



Experiment Code:
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\experiments\downstream_ocr

GNN Training/Augmentation Code:
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\src

Manuscript Path:
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\app\input_manuscripts\yajn
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\app\input_manuscripts\dense
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\app\input_manuscripts\circle_new

VLM Cache (we have already predicted the off-the-shelf VLM predictions)
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\app\tests\logs\downstream_ocr_vlm_cache

Note that I've change the dataset annotations a little bit, hence it's okay if some of the results table values do not exactly match the values of the current 5-fold experiment we have:
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\app\tests\logs\ocr_5fold_new
Hence only use this experiment result table values to do common sense verification if required. No strict checking. Ideally, after fine-tuning the GNN too, we should see a _steeper_ drop in both metric values, in the column (no-layout-postcorrection) over the rows pages finetuned 0,1,2,3 because hopefully the predicted layout would also improve with each page finetuned by GNN, along with improvement in text-line predictions of the OCR model.


Keep the both metric calculation unchanged. Do not make any other unnecessary changes to the experiment.


Right now, we have kept the experiment code as shared with the production app as possible. But as the production app at the moment does not support GNN finetuning, we will diverge the experiment code from the production when it comes to GNN finetuning. We will update the production app once the experiment is successful late on.

