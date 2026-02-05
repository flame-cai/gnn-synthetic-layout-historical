1) copy GT page xmls, and Images.

2) upload the images, as a new manuscript - USE TOOL. 
get gemini to use the GNN predicitons as grounding to perform OCR


3) upload the images, as a new manuscript - EASY OCR OUTPUT 
cd app/recognition
python recognize_manuscript_text.py complex_layout_remaining_12_pages


4) use the images to make gemini api call - GEMINI NO-STRUCTURE OUTPUT
cd downstream_eval
python gemini_ocr_no_structure.py complex/images


EVALUATE
cd downstream_eval
python evaluate.py



# DOCUFCN and SEAMFORMER
python gemini_transcriber_docufcn.py --images complex/images --xml complex/docufcn/page-xml
python gemini_transcriber_seamformer.py --images complex/images --xml complex/seamformer/page-xml


python gemini_transcriber_docufcn.py --images simple/images --xml simple/docufcn/page-xml
python gemini_transcriber_seamformer.py --images simple/images --xml simple/seamformer/page-xml


# RECOGNIZE (padding 10 for docufcn)
python recognize_manuscript_text_v2.py  --xml_folder ../../downstream_eval/complex/docufcn_easyocr_v2 --image_folder ../../downstream_eval/complex/images
python recognize_manuscript_text_v2.py  --xml_folder ../../downstream_eval/complex/seamformer_easyocr_v2 --image_folder ../../downstream_eval/complex/images


python recognize_manuscript_text_v2.py  --xml_folder ../../downstream_eval/simple/docufcn_easyocr_v2 --image_folder ../../downstream_eval/simple/images
python recognize_manuscript_text_v2.py  --xml_folder ../../downstream_eval/simple/seamformer_easyocr_v2 --image_folder ../../downstream_eval/simple/images