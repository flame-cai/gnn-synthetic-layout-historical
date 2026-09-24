# new
how does text recovery of the annotation tool working? as our OCR model will do bad, if I manually correct a small correction to a layout (adding a node to a text-line), will the tool be able to recover the text based on the text-line location? 
If I add a new text-line manually (because it was not transcribed in the new dataset labels), and then go to the read mode to annotate the unicode of that line, will we be able to recover the text of all other existing lines? Again, note that our OCR model will make bad predictions on the newar text, so we need to be able to rely on the text-line location to recover text

regarding Hitopadesa and Vetala, if 4 of 1,425 lines lose characters (vetala_0008/0009), then i guess ×3 upscaling works betterm because I want to minimize me manually annotatating (adding) missed characters..

and yes, fix the 34 in app/input_manuscripts (backed up first)



# Doubts
3. **CRAFT misses only marginal glyphs.** 24 lines had no node at all, every one a folio number or marginal mark; when you say they are rebuilt from the source baseline, is the respective conversion is the graph based format (with nodes and edges)? what do you mean by: 160 "characters missed at line ends are recovered from heatmap evidence"


on the sample subset, please check if min_distance 15 instead of 20 works better. For the two manuscript which are low resolution, instead of upscaling them (which can cause artifacts), can we instead use min_distance of 8 for such manuscripts? CRAFT also processes low resolution images faster so that is good. So try this approach on the subset first.

51 of 154 folio numbers
and marginal marks need a human look? mostly because of how the tool crops one- and two-node lines? what do you mean by this? I know that the fallback cropping (and thus the bounding polygons) might go wrong, but we care more about the gnn based format labels, and the unicode annotation than the bounding polygons. Please check what you mean by these "51" need manual annotation. 


# Answers to questions:
A1. Let's go with keeping the annotations in devanagari script, but also keeps a backup page xml files in a seperate unused folder which have everything else the same, but with the unicode labels in newa script.

A2. If the OCR model's character set does not have `॥`, i think we should replace precisely the 34 `॥` in the project's dataset with `।।`. For the new dataset too, write  `॥` as `।।`. Thus for all datasets we want to follow `।।` convention. Does this make sense?

A3. Please list which pages to look at which are examples of ink source being untranscribed. (**Untranscribed ink** exists on many leaves: later-hand marginal notes (`p006`), an interlinear insertion (`p004`), left-margin marks (Hitopadesa, Vetala), decorative end-of-text marks.) I'm planning to manually annotate the unicode of such line from the annotation tool by loading the pages. However, do we get the layouts annotations of such lines? or would i need to annotate the layout also manually?

# newar dataset
can you please study the below mentioned dataset, and check if the dataset can be converted to the format of our dataset in "app/input_manuscripts" which our annotation tool "app" produces (so please study how our annotation tool works too):

dataset: newar_sanskrit_dataset/dataset

Note that we mainly want the the unicode text annotations of text-lines from this new dataset. But before applying the unicode text-line annotation to the detected text-lines, we must ensure the layout predictions of our annotation tool on the mansucript images is right (and apply corrections where required use guidance from the to-be-converted dataset's layout labels). The layout annotations should be in format our annotation tool supports (but we can use the new datasets layout annotation to help with the conversion and fix mistakes in our annotation tool's automatic layout prediction failure cases)

We can also get the text-region labels perhaps along with the text-line labels from the to-be-converted new dataset's xml labels. Check if the new dataset has text-line level bounding polygons, and region level bounding polygons. We mainly want to use these polygons to label _our graph_ first, and then our bounding polygons will be derived from _our graph_ (like how our annotation tool does it). Only our tool cannot create the bounding polygons for some reason, we can use the new datasets text-line bounding polygons as is. So remember, the graph based text-line and text-region layout labels are what we want from the to-be-converted dataset's labelling format.
We won't need the word level annotation in the format we expect, but perhaps we can use them for conversion failure corrections or something like that..

Some mistakes our annotation tool layout predictions might makes:
- 1 text-line predicted as 2 text-lines
- 2 text-lines merged and predicted as 1 text-line
- CRAFT failed to detect all characters of a text-line, hence it was never detected..
- CRAFT failed to detect some characters of a text-line..
- CRAFT incorrectly detected something as a character..
- more such cases...

Also consider that the baseline labels of the to-be-converted dataset might also come handy if our CRAFT+GNN pipeline fails to detect the base line (we can't really do much if CRAFT doesn't detect all the characters in a text-line - then we will need to resort to using the baseline of the to-be-converted dataset with some offset (as our graph based "baseline" pass through the text-line, where as the new dataset's baseline might be about, below the text, or through the text..we don't know yet)). Do you get what I mean? We need to resort to using the new datasets labels "as is" as a final last resort (if there are layout failures in our annotation tool pipeline). As much as possible, we must get guidance from the new dataset's labels to convert the dataset to our own CRAFT+GNN pipeline conventions. 

Think about if the default hyperparameters of our pipeline (image resize, min_distance between character) will work for this dataset given the dataset image resolution. If not, tweak the hyperparameters. Different manuscript in the to-be-converted dataset might require different hyperparameters.

There are also some manuscript specific quirks we need to handle:
- images starting from "Hitopadesa Berlin" are good, but are a bit low resolution
- images starting from "Madhyama Svayambhupurana" actually contain two pages in one images (one below the other). So we might want do a horizontal slice and get 2 images from 1 images after conversion (and the page-xml lable coordinates will need to be handles accordingly.)
- images starting from "MS B Vetala" are good, but are a bit low resolution
- images starting from "p" are good, but contain meta text information at the bottom (in english), and also have a "scale/ruler" to the right of the actual mansucript. Both the ruler and the meta text are in white, and the background is black, so Projection profiles to differentiate between black and white should be able to crop them out (we don't want them). The cutting doesn't necessarity need to be dynamic for each such page. But the page-xml label coordinates will also need to be handles accordingly.


We should be able to load the converted dataset with our annotation tool with full functionality (so that after conversion I can manually check and make manual adjustments)

Run experiements on a subset first to get a feel of the problem, think carefully about how to do the conversion faithfully, and prepare a report which explains how to do the full conversion in the future. Ask me for any clarifications if required.
Do not read entire xml files into context as they might be too big and might pollute the context.


write relevant data processing/analyzing code/report in directory:
newar_sanskrit_dataset

Use conda environment "gnn_layout" of user kartik.


# chinese hindi
- Stage 1: base images
    - Geometry labels (min distance 3)
    - Gemini cutting the unicode region text to text-line unicode + human supervision
- Stage 2: photographed images
    - unicode text line labels remain the same, only geometry changes
    - can we transfer the geometry labels from base images to this using top left to bottom right ordering?

# Important 
Gemini should handle the table HTML labels and try to assign them to the graph based table (human will verify). CRAFT + GNN will detect each table element and connect it.

Gemini should also assign a text in figures unicode labels to the respective text-lines detected by CRAFT+GNN (they will detect the text in figures too - if they don't human can supervise). 

So human supervision can be adding and deleting nodes, edges, and also checking if text-region text is assigned to the text-lines properly.

Ideally we want human verification only for Stage 1. and we should be able to transfer the labels from stage 1 to stage 2 (text-line unicode annotations can be fully transfered, geometry can we inferred perhaps based on how the transformation is between base image and photographed images)




# PROBABLE PLAN

 Here's the full procedure. The key move is that you introduce a rotation yourself — and that transform you know exactly, so it's the one thing you
  can safely invert.

  Step 0 — Deskew first, before CRAFT or the GNN

  This has to come first because both models are orientation-sensitive: the GNN was trained on horizontal manuscript lines, so on a 90°-rotated page
  its output isn't trustworthy enough to estimate anything from.

  Estimate the text angle from CRAFT nodes alone, no GNN: build a histogram of the angles of each node's vectors to its nearest neighbours. Characters
  are packed far more densely along the baseline than across it, so the histogram's mode (mod 180°) is the text direction. Rotate the image by −θ and
  write that as images_resized/<page>.jpg. Then run CRAFT and the GNN on the deskewed image.

  180° stays ambiguous here — don't try to solve it now, resolve it in step 2 by scoring both and keeping the winner.

  Record θ. This is the point: you can't map base coordinates onto a folded photo, but the rotation you applied is a rigid transform you chose, so you
  can invert it exactly. Label on the deskewed image, then rotate node coordinates back by +θ if you want labels on the raw photo. Which you probably
  do — deskewing every page would strip out exactly the rotation robustness the photographed set was supposed to teach.

  Step 1 — Recover reading order from the photo alone

  Cluster the GNN's line components by horizontal overlap into columns (use each line's start x, not its full extent — on a bent page a column drifts
  sideways going down). Order columns left→right, lines top→bottom within each. That's an ordered line list for the photo with no reference to the
  base, which is what breaks the circularity.

  Step 2 — Align two sequences, don't match geometry

  From stage 1 you have an ordered base line list — region-aware, using MDPBench order across regions and top-to-bottom within. Each base line carries
  four things: its region anno_id, its Unicode, its length as a fraction of its region width, and its node count.

  Align the two lists with Needleman–Wunsch allowing gaps, scored on relative length, relative position within column, and node-count ratio.
  Line-length fraction is the workhorse — every paragraph ends short, so the sequence [1.0, 0.99, 0.43, 1.0, ...] is close to a page fingerprint and
  survives a few missing entries. Run the whole thing twice, at 0° and 180°, and keep the better score.

  Gaps are the point: they handle CRAFT missing a line under a fold, which a hard count-equality gate would just reject.

  Step 3 — Transfer, which is now trivial

  For each matched pair, base line i ↔ photo line j:

  ┌────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │            file            │                                     value                                      │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ _labels_textline.txt       │ j (renumbered) for every node in j                                             │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ _labels_textbox.txt        │ region_anno_id of base line i — the region label rides along the line          │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ _edges.txt                 │ MST over j's own nodes, i.e. what create_ground_truth_graph_edges already does │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ PAGE-XML TextEquiv/Unicode │ base line i's text, verbatim                                                   │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ _inputs_*.txt, _dims.txt   │ the photo's own CRAFT output                                                   │
  └────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

  Unmatched photo lines — fingers, desk, fold artefacts — get textbox = -1, no text, and a flag. Unmatched base lines mean CRAFT missed something;
  record as a defect for the human to add nodes.

  That's a complete gnn-format page. run_gnn_prediction_for_page reads layout_analysis_output/gnn-format/ in preference to gnn-dataset/, so writing
  those six files plus the resized image is enough for the tool to open the page, and generate_xml_and_images_for_page produces the PAGE-XML from the
  same graph.

  Step 4 — Gate it automatically

  Three numbers per page: the alignment score, the fraction of base lines matched, and the chars-per-node CV per region (threshold 0.20 — the one
  signal here I've actually validated, at 94% separation between a correct and a one-line-shifted assignment). Anything failing goes to review or gets
  dropped, and you report the yield rather than assuming it.

  What I'd be honest about

  Steps 0–3 are a design, not a tested result. The measured facts underneath them are: the CV check works (94%), 15 of 40 Hindi base pages are
  multi-column so ordering genuinely needs step 1, and the rotation is real and larger than my aspect-ratio proxy could see.

  The untested assumption is step 2 — that line-length sequences match reliably across a creased re-shoot. That's what I'd probe first, and cheaply:
  take 5 Hindi base pages and their 15 variants, run steps 0–1, and just look at whether the recovered column count and per-column line counts agree
  with the base. If they do, the alignment is straightforward. If the columns don't come out cleanly on the creased pages, you fall back to the 25
  single-column Hindi base pages, where step 1 is a plain (y, x) sort and none of this is needed.