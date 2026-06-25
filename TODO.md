

- remove backward compatibility bloat...
- long file name bug fix - shorten to 1,2,3

- check if telemetry change and heatmap assignment caused any unexpected effects.
- add padding for single characters...
- in yoooooooooosss line, if heatmap is not there for 40 percent of the line, but the baseline is extending, extendent the polygon!!!why is this not working?

- emergency layout change: override Coords in ReadMode and reread? what all should change? 
- check if telemetry is working..
- export in this format too: https://gemini.google.com/share/6d96e9a50411
- intra page finetuning overhaul..superfast?





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

