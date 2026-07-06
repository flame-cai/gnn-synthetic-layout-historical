
dynamic KV cache for OCR.
- put the big maps as the held out evaluation benchmark?
######################################

Paper update:
- Fine-tune only on 5 pages, test set on the rest.
- so test set:
    - ground truth layout + ground truth text
- inference on test-set:
    - ground truth layout + predicted text (after each page of finetuning, without and without manual layout edits)



Measure number of layout edits and time-taken per manuscript, per page.
- one key press is one key stroke
क्रियघटा ११ कौर्प्यां८डजौ१२याम्यतोमध्येन्येनवसंत्यथेंद्रककुभोवर्गाःस्युरोज
स्विनः ८६ अष्टावप्रमुखःस्वपंचमपराद्विन्घःस्ववर्गोन्ययुक्‌तष्टःकाकिणिकाग
जै८र्मिथइमायस्याधिकाःसौर्थदः श्वेतारत्ककपीतकृघ्मवसुधास्वादुःकटुस्तिक्तकाः
काषायाघृतशोणितान्नमदिरागंधाशुभा विप्रतः ८७सौम्यादिष्लवभूतलेविरचये
द्विप्रादिकोऽग्र्योखिलेनान्येषांनियमोत्रथत्रनिखिला:कुयुर्गृहंहृत्स्थिरं सद्मप्रघ्मकृ
तोमुखात्प्रथमतोवर्गादिवर्णोद्रमश्वेत्तद्दिग्गतमादिशेत्तुहयपैःशल्पंसुधीर्मध्यतः ८८
स्वभ्रंहस्तमितंखनेदिहजलंपूर्णंनिशास्येन्यसेत्प्रातरर्दृष्टजलस्थल्ंसर्दजलमध्यंत्वंस
त्स्फाटितं ज्ञात्वैवंनिखनेद्रृहाधिकभुवंनत्वाजलांतस्तरोयावद्वापुरुषस्ततःकपिशि
रस्तुल्याश्मभिःपूरयेत्‌ ८९ प्राक्‌साध्योज्जयिनीस्थलाद्यमदिशित्वाष्ट्रानिलाभ्यंतरा
त्सौम्येऽतोग्न्युदयादुदक्‌ध्रुवमुखादिग्मूढकेस्यान्मृतिः ९० गेहंमाधवपोषफाल्गुनन

implementation slices
36 - 43 - Page 10
80 - 85 - Page 17
101 - 105 - Page 22
105 - 110 - Page 23
112 - 113 - Page 24
114 - 116 - Page 25
123 - 126 - Page 28
126 - 131 - Page 29
135 - 141 - page 31
143 - 147 - page 33 (incomplete)



### TO DISABLE LOGGING KEYSTROKES:
Disable controls:
Backend: LAYOUT_EFFORT_LOGGING_ENABLED=false
Frontend: VITE_LAYOUT_EFFORT_LOGGING_ENABLED=false




#######

# UX TODO
- tag lines to exclude from training
- tab function should respect text-box annotations
- add a warning (once you save layout, make text corrections in read mode, then edit layout - then your existing corrections will get erased! automatically save a backup of the xml)

# TODO Typing fixes
First understand the grouping..consonents, dependent, independent..
No trickle down effects on other typing patterns. try to keep all other behaviour unchanged, but also try to find and maintain abstractions and invaraints.

1) ऋ

2) Ru is not working. We should be able to type र्नृ (r+n+R+u).. check with other rules..rnRu. If in doubt ask me. Do not change working of other rules. other examples: हृ (h+R+u), न्मृ (n+m+R+u) न्मृ


3)

Many times the user needs to only modify the a dependent vowel attached to a consonant. Right now, to do this, the user has to backspace to remove the wrong dependent vowel, and also backspace to remove the consonant. Then the user types the consonant again, with the right dependent vowel.

We want to change this typing UX such that the user can do this with only one backspace.
They should be able to quickly change ni to no, or ni to nou
लं to लें (backspace+e+M)
चै to चे (backspace+e)
so on..

can you please investigate if this issue is fixable by finding the right invariants and abstraction? how many corner cases? we don't want this change to change other typing behaviour in unexpected ways.

4) let us say I am editing the string: "म्कय"
when cursor is between क and य, and I hit backspace, the म automatically joins to the य. We don't want this. I just want to replace क, by hitting backspace, the pressing n to (न) to form म्न. So I just want to change म्क to a म्न easily by preventing म automatically joining to the य. 
This is just one example. There could be other such cases where I want to edit the string when the cursor is in the middle..
It is working as I want it for the string: 
त्यम, where I have the cursor between  य and म, and I want to change त्य to त्व.
Can you please find the discrepancy and reason for why it works for some examples and not for others, then can you find a robust fix with the right invariants and abstraction?
Example of what works well:
श्चैमके to श्वमके (starting from cursor between श्चै and म, backspace+backspace+v+a) -> hence we dont need to retype sh to get श्‌..that's good.













##########################


- Remove backward compatibility bloat and redundancies


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


# FUTURE TODO
- allow user to paint and make segmentation corrections in read more
- some layout changes should not trigger recognition model again.
- intra page recogntion finetuning overhaul..superfast?



# make demo video
- upload page
- document the image selection criteria, high resolution (CRAFT should be able to detect), reduce min-distance
- fix nodes edges (4000, 8 hyperparams)
- mark regions
- mark orientation
- recognize
- recognize with Gemini
- export as PAGE-XML.


# document the image selection criteria
- high resolution (CRAFT should be able to detect), reduce min-distance


_____________________






# Better GNN training
- Multi Task Learning, region, orinentation
- BIG TRAINING STEPS
- Write why MPNN training faster is a big advantage.
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

