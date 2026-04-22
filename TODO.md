- minimize the time taken from fine-tuning job finish, to checkpoint promotion
- understand checkpoint promotion and the "bank" verification. Do we compare performance of 1 page fine-tuned model vs 2 page fine-tuned model on a bank containing text-lines from page-1? that would give page 1 fine-tuned model an advantage!!


Okay now I have doubt about step 6. You said the real promotion gate is a separate non-regression verifier. After the sibling selector picks one checkpoint (which is now always best_norm_ED.pth by default), the app compares that candidate against the current active checkpoint on the bank of already-approved pages, using page CER and allowing only a small regression margin (regression_guard_abs = 0.005). But my doubt is, do we really compare performance of 1 page fine-tuned model vs 2 page fine-tuned model on a bank containing text-lines from page-1 (which we are calling "already-approved pages"? wouldn't that give page 1 fine-tuned model an advantage, because it's trained on the "test" data is a way?




Roll-out
- GNN hyper parameter search (better model, faster inference), reduce training time!!! MPNN+Algorithm best bet.
- GNN multi-task learning (text-boxes)..
- the CRAFT fine-tuning

- latest fine-tuned model is not working
- I am always seeing that the 0-page finetuned model is being used for recognition...is the model being loaded correctly 
- why do we fine tune two times (once immediately, and once after I )


- devanagari.pth
- what do we need to change to enable other scripts like bengali, grantha...