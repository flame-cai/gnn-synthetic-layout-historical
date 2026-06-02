okay great. Now can we focus on the following possible bugs?

If the whole line has no assigned heatmap boxes, the strategy falls back to a minimum-width band over the full baseline. 
This minimum-width band is too small. To fix this, perhaps
example:
C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\app\input_manuscripts\my_manuscript_testing_2\layout_analysis_output\image-format\233_0002\textbox_label_0\line_22.jpg




If the missing character is only at the leading or trailing end, and there is no heatmap component there, the polygon may not fully extend to that manually added endpoint unless padding/fallback happens to cover it. Perhaps we can use the average contour height of that text-line to extend upto the newly added nodes?
We observe this bug in the below example:
C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\app\input_manuscripts\my_manuscript_testing_2\layout_analysis_output\image-format\233_0002\textbox_label_0\line_5.jpg



# demo video
- upload page
- fix nodes edges (4000, 8 hyperparams)
- mark regions
- mark orientation
- recognize
- recognize with Gemini
- export as PAGE-XML.


# production
- do not have red edges, only color code the nodes!
- line orientation annotation smoothers
- recheck orientation GUI
- heatmap joining problem - when the heatmap is bad, we can't do much..
    - more config on new manuscript page
    - graph corrections don't really work when the heatmap is wrong..check this if it's working
    - make node additions and deletions affect the PAGE-XML Coords preparation. I thought his was working previously...
    - manual updating bounding polygon in read mode? that will update PAGE-COORs, and will do the unwrapping again

# document the image selection criteria
- high resolution (CRAFT should be able to detect)


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
- Multi Task Learning
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

### 
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
