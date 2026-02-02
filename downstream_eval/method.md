1) copy GT page xmls, and Images.

2) upload the images, as a new manuscript - USE TOOL. GEMINI STRUCTURE OUTPUT


3) upload the images, as a new manuscript - EASY OCR OUTPUT 
cd app/recognition
python recognize_manuscript_text.py complex_easyocr_8


4) use the images to make gemini api call - GEMINI NO-STRUCTURE OUTPUT
cd downstream_eval
python gemini_ocr_no_structure.py complex/images
