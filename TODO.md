FIX THIS:


In this production application (not the research harness), when the annotated text-lines (after the user makes corrections by adding and deleting edges/nodes, assign text-line orinetations, and text-region labels), we use the annotations, along with the heatmaps, to create the PAGE-XML Coords.

Ideally this process uses Heatmaps, where the location of each character of the image is hot in the heatmap. But sometimes the heatmap fails to do this for some messy characters. So we expect the user to do add these nodes manually.

However, sometimes if the whole line (consisting of one or more characters) has no assigned heatmap boxes, the strategy falls back to a minimum-width band over the full baseline. This minimum-width band is too small. 

I want your help in fixing this precisely. To fix this, perhaps can do adaptive local binazrization of that region of the image, and then fit a bounding rectangle plus some padding around the binarized part which would ideally be the character(s) which the heatmap was not catching.

Hence please study the code (in the production app with GUI), understand the code, and make precise changes to fix this.


- faster saving
- d and hover deletes nodes too
- Orientation
    - no skipping
    - faster, no lag
    - single characters not being able to orient..






check if the docs are in sync with the app






# make demo video
- upload page
- document the image selection criteria, high resolution (CRAFT should be able to detect), reduce min-distance
- fix nodes edges (4000, 8 hyperparams)
- mark regions
- mark orientation
- recognize
- recognize with Gemini
- export as PAGE-XML.


# production

- orientation GUI doesn't work the first time?


# document the image selection criteria
- high resolution (CRAFT should be able to detect), reduce min-distance


_______________

- do rigourous testing of the Application GUI
- make the pre-commit gates faster..
- faster strategy..
- GUI line orientation decision - new data collection type
- update docs and sync up the code base..
    - rewrite the strategy docs according to this architecture
    - write a template on how I should write out ideas for new proposed strategy
- Keep old strategies as backup.
- annotate all 481 pages at layout level and prepare a nice dataset






# Better GNN training
- Multi Task Learning, region, orinentation
- BIG TRAINING STEPS
- BIG MODEL - MPNN
- BIG DATA
- CREATE A BENCHMARK DATASET - all 481 pages
    - scrape_images.py markdown text file
- Synthetic data generator inspired by
    - colab notebook (for font rendering)
    - curved and synthetic lines
    - gnn format layout generator
    - it should generate data in the same format as 'eval_data'.

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

# Gemma finetuning:
Gemma Finetuning with and without visual grounding.


# Misc
- Gemini Magic Click button - keep it for experimental legacy purposes.
	- Make a tutorial how to use YouTube video..
		- how to install
		- how to run

