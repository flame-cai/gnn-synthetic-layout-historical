import json

def convert_ocr_json(input_path, output_path):
    """
    Convert predicted OCR JSON into ground-truth compatible format.
    """

    # Load input JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []

    # Iterate through OCR details
    for item in data.get("details", []):
        page_number = item.get("page_number")
        content = item.get("ocr_content", "").strip()

        converted.append({
            "page_number": page_number,
            "content": content
        })

    # Write output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)

    print(f"Conversion complete. Output written to: {output_path}")


if __name__ == "__main__":
    input_json = "no_structure_results.json"     # your input file
    output_json = "no_structure.json" # desired output file

    convert_ocr_json(input_json, output_json)
