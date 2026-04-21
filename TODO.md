Refresh OCR button not visible in layout mode.


2) Because not all buttons and elements are visible in each mode and the state, sometimes new elements appearing or disappearing can change the GUI a bit with jittery changes. Please don't let this happen. 

3) please do a Maintainence pass to check different paths user can behave using the GUI, and how the backend will handle it and try to catch edge-cases. Use the good ENGINEERING_DOCTRINE.md, and make the code better written, but ensure the functionality does not change.



Roll-out
- GNN hyper parameter search (better model, faster inference), reduce training time!!! MPNN+Algorithm best bet.
- GNN multi-task learning (text-boxes)..
- the CRAFT fine-tuning

- latest fine-tuned model is not working
- I am always seeing that the 0-page finetuned model is being used for recognition...is the model being loaded correctly 
- why do we fine tune two times (once immediately, and once after I )


- devanagari.pth
- what do we need to change to enable other scripts like bengali, grantha...