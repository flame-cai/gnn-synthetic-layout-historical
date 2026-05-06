- does this help? (only for devanagari)
    - reading order always left to right
    - always clockwise for circular layouts
    - make this configurable according to the script.
    - the GUI should allow user to just decide.

pre-commit new checks - implement plan
C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\docs\exec-plans\proposed\two-new-checks.md










- annotate data
- prepare dataset with the same format as eval_dataset
- a new test using circular_dataset
- LLM + verifier combo attack at:
    - ENSURE ALL THESE CHANGES MAINTAIN INVARIANTS and FROM ABSTRACTIONS. A straight line, is a specific case of a curved line and a cut circle.
    - how we crop the line using the graph polyline (but we must keep the heuristic metrics)
    - how to cut, and how to unroll a circle
    - how to unroll a curved "s" type line (metric tensor which uses the GNN polyline, but also keeps the heuristic logic)
    - can we unrolled, by simply rotating the cropped square wrt to the polyline, and then do the square joining algo..
    - when a circle is being unrolled, or if we have a vertical line - how do we know the orientation? use the uncertainty of OCR model to detect the orientation? this is actually reading order detection!!

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
    - curved and circular synthetic lines
    - gnn format layout generator
    - it should generate data in the same format as 'eval_data'
    - appearance level augmentations, and layout level augmentation

















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
