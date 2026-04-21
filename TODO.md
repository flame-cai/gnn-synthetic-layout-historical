Refresh OCR button not visible in layout mode.
When in recognition mode for Page 1, and the corrections are done, and I click commit and next, or go to next page, I should go the layout analysis mode of the new page. Make sure this is robust and doesn't introduce any new bugs.
Rare Character Button less ugly and in theme.


1) When we display this text: "Make corrections in Layout Mode before using Recognition Mode." in code:
const applyPageWorkflow = (payload = {}) => {
  pageWorkflow.state = payload.state || 'missing_page_xml'
  pageWorkflow.label = payload.label || 'Layout Analysis Not Done'
  pageWorkflow.hint = payload.hint || 'Make corrections in Layout Mode before using Recognition Mode.',
We don't want to show the 'Recognize Page', 'Commit', 'Commit and Next' Buttons. Maybe in the same hint, we have have a button which says "Go to Layout Mode"

2) Because not all buttons and elements are visible in each mode and the state, sometimes new elements appearing or disappearing can change the GUI a bit with jittery changes. Please don't let this happen. 

3) Handle this bug in the app in the GUI and backend orchestration: 
Take this user behavious - User Makes Layout corrections in Layout Mode, then makes manual corrections to the recognition model output in recognition mode, then goes back to Layout Mode (and makes no layout changes), and then comes back to recogntion mode. Right now, when this happens, the app runs the recognition again with the latest model. This is not desirable. If the user has made no changes in the Layout mode, do not recognize again, load the existing user-corrected text!. If the user want to use the latest finetuned model, they can manually click the button "Refresh OCR" right?


Roll-out
- GNN hyper parameter search (better model, faster inference), reduce training time!!! MPNN+Algorithm best bet.
- GNN multi-task learning (text-boxes)..
- the CRAFT fine-tuning

- latest fine-tuned model is not working
- I am always seeing that the 0-page finetuned model is being used for recognition...is the model being loaded correctly 
- why do we fine tune two times (once immediately, and once after I )


- devanagari.pth
- what do we need to change to enable other scripts like bengali, grantha...