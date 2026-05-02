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