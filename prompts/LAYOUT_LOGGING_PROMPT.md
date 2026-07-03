in the production OCR app, we have two parts, layout mode and read mode. Layout mode is used to make manual corrections

a) Can you please help me log and measure how much time the user takes making edits in Layout Mode?
We want to do a study on how much time it takes to go from a predicted layout to a ground truth layout (which is saved before we go to Read Mode)

So if it's a new manuscript, or even if the user revises a existing layout, we want to measure time taken by user per page: from first edit user does, to last edit user does on the page. If there is only one edit, we will consider time taken as 2 secs. We also want to measure the number of edits:

What I mean by an edit:
- each left click to add node will be one edit
- each keypress hold and hover would count as one edit, irrespective of number of nodes, edges, text-lines hovered on: 
"a" to add edge, "d" to delete nodes or edges, "e" to label text-line regions, "q" to label text-line orientation
For this kind of edit, also measure time is keypress is held.

If a user reopens an existing manuscript, and makes new changes to layout, we want to update the time, and edit counts.

Please have a dedicated logging json for this.

b) In the same dedicated logging json, move the existing layout based logging aswell:
- nodes added
- nodes deletec
- original number of nodes
- final number of nodes
- edges added
- edges deleted
- original number of edges
- final number of edges
- text-lines region labeled
- text-line orientations labeled (manually, not the default orientation assumed)

c) In the same logging sections, please also save the time-taken for processing the page: everything which happens between going from Layout Mode to the manuscript being ready for Read Mode OCR model (with text-lines images). If the user revisits an existing manuscript layout, and makes an edit, and saves it again, do not the consider the processing time for that page twice (just use the processing time of the latest layout revision)

__

Please take care to handle ambiguities and tricky edge cases, unexpected effects in the scenario where the user re-annotates an existing manuscript layout.

This logging sections will have a log per page, per manuscript, and will also have a cumulative summary for all pages of a manuscript. This will help us log how much human effort is spent in the layout mode.

Do not care about backward compatibility, as this is an additive change.

This logging implementation should not change any of the applicaiton functioning. It is a precise additive logging mechanism, which is enabled by default, but can be disabled.

Before implementing, please study the code, understand the requirements and ask my for clarifications if required.