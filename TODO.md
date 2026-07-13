# PLEASE DO NOT IMPLEMENT THESE UNLESS SPECIFICALLY REQUESTED

dynamic KV cache for OCR.
- put the big maps as the held out evaluation benchmark?

######################################


sanitize all paths and windows specific things, make them relative:
C:\\Users\\intro\\Documents\\Projects\\

# Digitization

## Recheck these
व्युहिश्र्वेत्‌=१
छद्रवति=४



implementation slices
36 - 43 - Page 10 
80 - 85 - Page 17
101 - 105 - Page 22
105 - 110 - Page 23
112 - 113 - Page 24
114 - 116 - Page 25
123 - 126 - Page 28


126 - 131 - Page 29
135 - 141 - page 31
143 - 147 - page 33 (incomplete)


# UX TODO
- tag lines to exclude from training
- tab function should respect text-box annotations

#### TODO Typing fixes
First understand the grouping..consonents, dependent, independent..
No trickle down effects on other typing patterns. try to keep all other behaviour unchanged, but also try to find and maintain abstractions and invaraints.

1) ऋ, द्भ, dyu



# make demo video
- upload page
- document the image selection criteria, high resolution (CRAFT should be able to detect), reduce min-distance
- fix nodes edges (4000, 8 hyperparams)
- mark regions
- mark orientation
- recognize
- recognize with Gemini
- export as PAGE-XML.





_____________________






# Better GNN training
- Multi Task Learning, region, orinentation
- BIG TRAINING STEPS
- Write why MPNN training faster is a big advantage.
- BIG MODEL - MPNN
- BIG DATA
- CREATE A BENCHMARK DATASET - all 481 pages
    - scrape_images.py markdown text file
- Synthetic data generator inspired by
    - colab notebook (for font rendering)
    - curved and synthetic lines
    - gnn format layout generator
    - it should generate data in the same format as 'eval_data', and the gemini JSON format for image generation..
    - https://github.com/GbotHQ/Blender-3D-document-rendering-pipeline

### Misc
- fix GNN loading model - state_load_dict
- end to end synthetic data generation, finetuning and evaluation (to improve any part of the pipeline! )
- GNN augment + synthetic data -- setup experiment with verifier
- GNN hyper parameter search (better model, faster inference), reduce training time!!! MPNN+Algorithm (no-algorithm) best bet?? faster data preparation
- GNN multi-task learning (text-boxes)
- the CRAFT fine-tuning
- latest fine-tuned model is not working
- I am always seeing that the 0-page finetuned model is being used for recognition...is the model being loaded correctly 
- why do we fine tune two times (once immediately, and once after I )
- devanagari.pth
- what do we need to change to enable other scripts like bengali, grantha...
- Remove backward compatibility bloat and redundancies
- allow user to paint and make segmentation corrections in read mode
- intra page recogntion finetuning overhaul..superfast?
- TO DISABLE LOGGING KEYSTROKES:
    Disable controls:
    Backend: LAYOUT_EFFORT_LOGGING_ENABLED=false
    Frontend: VITE_LAYOUT_EFFORT_LOGGING_ENABLED=false

# Gemma finetuning:
Gemma Finetuning with and without visual grounding.


# Misc
- Gemini Magic Click button - keep it for experimental legacy purposes.
	- Make a tutorial how to use YouTube video..
		- how to install
		- how to run

