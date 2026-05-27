- the proposed is too local..horizontal lines are treated as curved lines..



- drawing reading order annotation lines dont work on an entire paragraph it only works on one line at a time
- the research harness is untouched right?
- update gemini prompt
- don't touch the research harness
- the research harness proposed and benchmark is independent from the production app strategy.
- apply the reading order annotation to the Gemini recognition pipeline too..
- does production have base line normalization?

__________

can you please carefully prepare a plan to promote the current benchmark strategy in the research harness to production app?

Make sure to pay to attention to the following:
- The productions app's control flow should remain robust, and no unexpected effects should happen. When the user will click on a circular text in read mode. The active learning should still work, but should appropriately use the text-line image prepared by the new strategy (current research harness benchmark)
- Use the ENGINEERING_DOCTRINE.md to modularize well, and find abstrations and invariants where possible. The current research benchmarks strategy and it's production app implementation should have two steps: a) PAGE-XML Coords preparation, b) Preparing text-line images for the OCR model inference and fine-tuning (unwrapping, median color background..). Do you agree with this modularization? 
- make sure to take note of the all the configurations of the pipelines. The strategy to be promoted has a binarization threshold 0.45 instead of 0.509 for example.
- how does the research harness benchmark strategy handle ambiguous text-line orientation? As we know the script is devanagari, the reading order is left-to-right, and circular text is always left to right. Do you think we can handle this ambiguity if we all the user to optionally annotate the reading order of text-line in layout mode? For example, let us say we have two vertical text lines. However both these text lines are not of the same orientation. One goes from top to bottom, and other goes from bottom to top. In other worlds, the orientation of these text-lines is such that each line is 180 degree rotated with respect to the other. If the user can annotate the reading order of these vertical text-lines in the GUI, do you think this ambiguity can be resolved optionally if such annotations are made in layout mode? If not annotated we apply script specific defaults. We should have a new shortcut for reading order annotation mode. When user will press this shortcut, and cut the text-line in a particaular way, they will determine the reading order of that line. For example, if I press the shortcut, and cut a vertical line from left to right, it means the orientation of the text-line is (top to bottom). If I cut it it right to left, it means orientation is bottom to top. For most horizonal line, we should cut them by pressing shortcut and moving mouse bottom to top (thus determining their orientation as left to right -  the defaults). Please try to carefully understand what I'm saying here, and ask me for clarifications if needed. Another thing about the reading order annotation:

Add reading-order annotation state in layout mode. Use shortcut O; the user draws a cross-line cut. Interpret stroke vector (dx, dy) as intended reading tangent (-dy, dx), matching:
vertical left-to-right cut -> top-to-bottom
vertical right-to-left cut -> bottom-to-top
horizontal bottom-to-top cut -> left-to-right

Important: this should work on slant, curved lines too..the cut tangent reading order intuition is correct. But please refine

- Handle trickle down effects, and do not break upstream or downstream code. Look at this change from a system POV.

Please ask me for any clarifications if required.



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
