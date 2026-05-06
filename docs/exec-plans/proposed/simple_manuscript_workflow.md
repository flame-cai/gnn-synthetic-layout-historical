
We want to allow the user to switch between the following modes of annotation depending on the target manuscript.

Normal Mode 
The user will first annotate the layout, then annotate the text. Clicking on next page (or going to a new page) will take the user to the layout mode of the next page. Improve future can be ENABLED here (by default), but user may turn it off. 

Layout Only Mode (Annotating Layouts without recogntion)
This is for manuscripts with complex layouts, where the user will prefer to annotated the layouts of all pages first, and then come back later to annotate the text. However they might occasionally go to Text Review Mode for a couple of pages. Going to next page after save, or using the GO TO dropdown to go to another page, will by open the Layout Mode for that page. Improve future reading will need to be DISABLED here, as humans wont be annotating text, and recognition won't be running in the background.

Recognition Only Mode (For simple layouts)
This mode is for manuscripts with simple single column layouts where almost no layout corrections are required (or for manuscripts whose layouts have been annotated in the Layout Only Mode previously). In this mode, the user will want to directly work in text review mode and annotate the text. However, on rare occasions, they will want to make slight changes to the layout for a rare page. We should support that. Going to next page after save, or using the GO TO dropdown to go to another page, will by open the Text Review Mode for that page.
Improve future can be ENABLED here (by default), but user may turn it off. So this mode will essentially in the background, go to layout mode, assume no changes are required, and then go to text review mode (because layout mode is required before text review mode)

Perhaps can we simplify this, because if the user clicks on save and next page from layout only mode, it does go to next page in the layout mode itself. So this is like the layout mode. So I guess all we need is a "skip layout mode" toggle (with help info showing that this should be used if user has previously annotated all the layouts, or if the layouts are too easy and can be skipped.) This one toggle will essentially enable recognition only mode?
What do you think?

Please study these proposed workflow (or just a simple new skip layout mode toggle button) in the context of the entire application as a whole being a system, and try to find gaps and bugs which could arise. Please refer to the relevant docs (AGENTS.md, ENGINEERING_DOCTRINE.md, EVAL.md) and code. 
