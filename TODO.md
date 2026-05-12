Please implement the following plan, which will implement step 4 of the following research harness:

The research harness will enable us to combine the generative capabilities of LLMs with external verifier metrics to perform step by step evolutionary search in python code space. 

This harness can we adapted to improve any part of the application pipeline using the external verifier metrics (See VISION.md). However, the current part of the pipeline this research harness is aimed at improving is how this application segments text-lines (using the resized_images, heatmaps and the respective predictions of the GNN i.e the <Baseline points="x1,y1 x2,y2 .."/> in page-xml). In other words, we currently want to improve the part of the pipeline which converts the GNN predictions <Baseline points="x1,y1 x2,y2 .."/> in page-xml) to the text-line images which are fed to the downstream OCR model for fine-tuning or inference. The application already has a text-line segmentation strategy. We will call this the "benchmark_strategy". However this strategy is flawed and works only for horizontal text-lines (not circular, curved, or vertical lines). To fix these flaws, we have implemented a "proposed_strategy".


Step 1 (IMPLEMENTED): Understand the code, and precisely scope what it is that needs ablations to be iteratively improved.

Step 2 (IMPLEMENTED): Setup 3 external verifier precommit checks. Each verifier will have 2 runs, ablating and comparing "proposed_strategy" vs "benchmark_strategy". Each of the 3 external-verifier pre-commit checks should use the exact same implementation for each ablation strategy.

Step 3 (IMPLEMENTED): Implement the first iteration of the "proposed_strategy", with the aim to handle circular, curved and vertical lines better, while maintaining performance on horizontal lines.

Step 4 (TO BE IMPLEMENTED): Setup strategy promotion mechanism and configuration (from "proposed_strategy" promotes to "benchmark_strategy") if pre-commit checks passes.

All strategy docs live here:
C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\docs\pipeline-improvement\text-line-segmentation\


Please update VISION.md
PLease write a EVAL.md explaining the evaluation harness (each of the three tests, what is success criterai, the config, and more).

Both VISION.md, and EVAL.md should have self-contained information for future agents to adapt this harness to improve another specific part of the pipeline (after specifying what the inputs and outputs are.)

I see the workflow as follows:
1) I suggest new proposed_strategy
2) LLM agents implement, and update code
3) I click commit
4) Commit passes, if all 3 pre-commits checks pass
5) proposed_strategy promoted to benchmark strategy (docs updated in C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\docs\pipeline-improvement\text-line-segmentation, configs updated, other changes to make everything self contained. We want to keep all historical code as backup and documented.)
6) I suggest a new proposed_strategy
and so on..

Based on this, implement this plan:
C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\docs\exec-plans\proposed\circular-text-04-promotion-and-docs.md




# stray thoughts

- GUI should be able to annotate line orientations (new supervised task)

- please prepare a doc for comparing proposed_strategy vs benchmark_strategy, update vision.md, and eval.md...not just for segmentation, but also for every other part of the pipeline.


- let's implement step 4, and also write a document which explain the proposed_strategy vs benchmark_strategy pre-commit check harness, and how proposed_strategy gets promoted to a benchmark strategy of the next iteration. Keep old strategies as backup.

So for each iteration (after a successful commit after a precommit check has passed) I want to have an ITERATION.md file which will contain the following sections:
- the architectre of the current benchmark method - step by step blueprint of the pipeline
- what direction we want iterate this, new ideas, what changes to make
- use the blueprint doc!!!!!!!!


PROPOSED PLAN ITERATION:
- WHY ARE WE NOT CROPPING A CONTOUR LIKE THE BENCHMARK STRATEGY. WE WANT LOCAL TANGENT CONTOURS, WHICH ARE UNWRAPPED, NOT BANDS WHICH GET TEXT FROM ADJACENT LINES.
- short curved lines?
- full generalization? don't export to old strategy
- compare still with the og benchmark for now.














CIRCULAR LAYOUT TODO
- annotate manuscripts in Tantra and alaṅkāra
- fix vertical lines recognition bug
- enable annotation, recognition (and fine-tuning) for circular and curved lines 
- iteratively finetune both GNN and EasyOCR...measure hopefully rapid reduction in human effort
- annotate all 481 pages (text regions and text..)
- synthetically generate text-lines images (and gt annotations, with different white noise levels, texture, font) such that the text-lines are vertical, horizontal, curved, circular. GNN can be trained to detect the text-lines even if they are curved af. Then comes the magic - how to process curved lines, and iteratively finetune EasyOCR to get the recognition model working? we first need to train the GNN. We want a curved line recognition strategy which gives fast finetuning improvements, as quantified by the external evaluator. We have ground truth data as this is gonna be synthetic.


- try superhero skills
step 0 finish the fine-tuning module..(clean up, update readme and installation guide, and docs)
step 1 annotate real data and create synthetic data on the side.
step 2 evolutionary vibe code to optimize for iterative fine tunes accuracy - as verifier. Fix layout to be perfect for this test. what's gonna be vibe coded would be the line processing strategy of any circular lines
step 3 ask shagun to annotate and be coauthor
- - automatically delete models finetuned as precommit checks. keep the base model.


TODO
- vertical lines
    - use uncertainty
    - give option in GUI to decide the line orientation..
    - collect this data too!!
- curved and circular lines (metric tensor!)
    - it just a converter from one map to another 
        - simple (2x scale up)
        - dynamic (earth to map) 
    - cut at topmost point if closed.
    - the metric tensor we used is tightly linked to the OCR model we finetune!
    - we know the gnn based polyline
- synthetic data generator inspired by
    - colab notebook (for font rendering)
    - curved and synthetic lines
    - gnn format layout generator
    - it should generate data in the same format as 'eval_data'.












____________________________________
OTHER TODO
- fix GNN loading model - state_load_dict
- end to end synthetic data generation, finetuning and evaluation (to improve any part of the pipeline! )
- annotate eval_data for text-regions
- annotate input manuscript with text-regions.
- GNN augment + synthetic data -- setup experiment with verifier
- GNN hyper parameter search (better model, faster inference), reduce training time!!! MPNN+Algorithm (no-algorithm) best bet?? faster data preparation
- GNN multi-task learning (text-boxes)
- traditional vs Gemma 4 fine-tuning comparison..
- 
- the CRAFT fine-tuning

- latest fine-tuned model is not working
- I am always seeing that the 0-page finetuned model is being used for recognition...is the model being loaded correctly 
- why do we fine tune two times (once immediately, and once after I )


- devanagari.pth
- what do we need to change to enable other scripts like bengali, grantha...
