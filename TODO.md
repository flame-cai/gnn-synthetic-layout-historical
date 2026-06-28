
dynamic KV cache for OCR.
- put the big maps as the held out evaluation benchmark?
######################################





can we speed up step 3? and also control for this misleading s/2 offest (because we don't offset heatmap component center?):
3) For each heatmap component, find the nearest baseline by distance from the component center to the baseline polyline. For point baselines, this is just Euclidean distance to the single point. See [local_polygons_stable_unwrap.py (line 2466)](C:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/line_segmentation/local_polygons_stable_unwrap.py:2466).
this offset also might cause confusion for text-lines with different orientation. I think for this assignment, we should not be using offsetting baselines. I still can't believe that we are assigning heatmap components based on offseted baselines. 


##########################

- speed up the layout analysis without affecting function
- Remove backward compatibility bloat and redundancies
- understand short line axis alignment
- add padding for single characters..
- long file name bug fix - shorten to 1,2,3
- Read mode should allow painting and editing the text-line polygons. This will only change the Coords of that specific line, it will then be unwrapped, and text-will be recognized only for that line! this is a precise vertical - meant to handle outliers. what all should change? 
- allow user to draw polygon over graphics
- check if telemetry is working correctly (CER edits required should drop with each fine-tune)
- intra page finetuning overhaul..superfast?

# Export Save Format:
    - DocOmniBench Format
    - PAGE-XML Support from 13 to 19, supporting Graphic and Table annotation. 
    - Diffusion Model Prompt format:  https://gemini.google.com/share/6d96e9a50411

# Setup Coords Segmentation Eval Research Harness:
    - Normal lines
    - conjusted lines in the map
    - single characters, page numbers
    - missing heatmap
    - extra heatmap

# ANNOTATION RULES:
- A text block should contain only text that naturally belongs together and can be read in one clear and unambiguous order.





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

# Gemma finetuning:
Gemma Finetuning with and without visual grounding.


# Misc
- Gemini Magic Click button - keep it for experimental legacy purposes.
	- Make a tutorial how to use YouTube video..
		- how to install
		- how to run

