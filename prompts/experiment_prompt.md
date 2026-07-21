
You are an expert in deep learning research, specializing in OCR, Document Layout Analysis, text-line segmentation, Handwritten Text Recognition, and Multi-Model Large Language Models and experiment hygience (folds, data-splits, seeds etc) You also think like an expert researcher and a functional programmer, and write standardized code with the right abstractions and invariants to do this experiment. 
Can you please study the experiment we are doing in this repo, where we reuse production "app" code as much as possible to study various ablations of OCR methods such as:
Gemini end to end
Gemini (without layout correction)(this is disable right now)
Annotation tool (with and without Layout correction) and Annotation tool (with 0/1/2/3 Page fine-tuning)
Experiment code:
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\experiments
Production app code:
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\app
Each experiment run prepares a report like the following test run which we ran for 1 fold:
C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical\app\tests\logs\downstream_ocr_change2_1fold
using the command:
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-methods   --manuscript-root app\input_manuscripts\yajn
  --manuscript-root app\input_manuscripts\dense   --manuscript-root app\input_manuscripts\circle_new
  --output-root app\tests\logs\downstream_ocr_change2_1fold   --fold-id fold_1
  --method-id vlm_e2e   --method-id annotation_tool_e2e
  --method-id annotation_tool_gt_layout   --method-id annotation_tool_pred_layout_ft_1
  --method-id annotation_tool_gt_layout_ft_1   --method-id annotation_tool_pred_layout_ft_2
  --method-id annotation_tool_gt_layout_ft_2   --method-id annotation_tool_pred_layout_ft_3
  --method-id annotation_tool_gt_layout_ft_3
We will be doing the experiments with the following datasets,  but the code to be organized in a way that it can easily apply do other such datasets (manuscripts) with the same directory and labelling format:
".\app\input_manuscripts\yajn"
".\app\input_manuscripts\dense"
".\app\input_manuscripts\circle_new"
Please do not flag experiment hygiene points below. These are accepted:
Repaired geometry of Ground Truth (GT) and predicted polygons is an accepted part of the experiment protocol. Production regenerated crop polygons from GT baselines are also an accepted part of the protocol. So do not flag these. 
In this experiment we do not use text-region labels anywhere 
Failure categorization is not fully consistent. 
Live-run provenance is insufficient for exact reproduction: The live manifest recorded commit edbfcd0 with a dirty worktree, but it did not record the dirty patch. That exact code state cannot be reconstructed from the manifest. 
Final reports can silently compare unequal page sets.  
Ignore the failing assertion: The assertion which compares local_polygons_v1 and local_polygons_stable_unwrap_v1 strategy coordinates before any fine-tune sample collection runs. This is irrelevant.
# Very Important points:
The experiment is independent of the actual production app and the rest of the code, although we reuse code from the production app where ever possible

changing the code in the production "app" is forbidden. 

Have a robust experiment methodology. For each experiment fold, for all methods being compared, the pages we are evaluating on should be the same. Do the data splitting carefully. 

Maintain the ability to add more methods (other than Gemini) to compare later in the future, so keep the code modular. We should also be able to add more manuscript for evaluation in the future. 

We already have gemini API key setup in *.\app.env), and would be adding other API keys in the future.
































Okay great, now can you please help me update the experiment in the following way:
Because Gemini API calls cost money, we want to use Gemini End-to-End (and also Gemini with layout corrections if it is enabled in the future) only once to get predictions of all pages of all three manuscripts "dense", "circular_new", and "yajn", and track the evaluation metric on a page by page level. This will allow us to only get the predictions of each provider like Gemini/Sarvam/openai/etc only once per page (with the current 3 retires), and then the experiment run command will simply reuse these "pre-predictions" as per the respective data splits of the fold.
Hence I want you to do the following:
in app/.env we have added other API keys. So please extend the code with good experiment hygiene (using the exact input and output handling) to other VLM providers Sarvam, Openai, Claude, and Deepseek, along with the Gemini. This will allow us "pre-predict" on all pages, of all three manuscripts, and then use these in the experiment run.
GEMINI_API_KEY="XXXX"
SARVAM_API_KEY="XXXX"
OPENAI_API_KEY="XXXX"
CLAUDE_API_KEY="XXXX"
DEEPSEEK_API_KEY="XXXX"
create a dedicated CLI command which will do VLM inference for various providers Gemini/Sarvam/openai/etc) to get the "pre-predictions" which the current experiment CLI can reuse (as per the respective data splits and folds)

modify the current experiment CLI command (which runs the experiment for annotation tool and Gemini/Sarvam/openai/etc) such that it uses these "pre-predictions", and the respective data splits in each fold, to calculate the metrics and the reports. The current experiment CLI should allow each VLM provider as input (Sarvam, Openai, Claude, and Deepseek, Gemini.)

DO NOT try to support backward compatibility. 
Do you think this makes sense? please ask me for any clarifications if required