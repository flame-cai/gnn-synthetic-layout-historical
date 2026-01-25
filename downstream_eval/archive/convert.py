import json
import re
import argparse
import sys
import os

def convert_sanskrit_md_to_json(markdown_text):
    """
    Parses markdown text containing page numbers and Sanskrit text,
    removes all whitespace, and converts to JSON list format.
    """
    # 1. Split the text into sections based on "Page No. X"
    pattern = re.compile(r'(Page No\.\s*\d+)')
    parts = pattern.split(markdown_text)
    
    json_output = []
    current_page_num = None
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Check if this part is a Page Header
        header_match = re.match(r'Page No\.\s*(\d+)', part, re.IGNORECASE)
        
        if header_match:
            current_page_num = int(header_match.group(1))
        else:
            # This is the content associated with the previously found page
            if current_page_num is not None:
                # Remove ALL whitespace (spaces, tabs, newlines)
                clean_text = "".join(part.split())
                
                entry = {
                    "page_number": current_page_num,
                    "content": clean_text
                }
                json_output.append(entry)
                
                # Reset current page
                current_page_num = None

    return json_output

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Convert Sanskrit Markdown transcriptions to JSON list format."
    )
    
    parser.add_argument(
        'input_file', 
        type=str, 
        help='Path to the input markdown file (e.g., input.md)'
    )
    parser.add_argument(
        '--output', 
        '-o', 
        type=str, 
        default='output.json', 
        help='Path to the output JSON file (default: output.json)'
    )

    args = parser.parse_args()

    # 2. Validation
    if not os.path.exists(args.input_file):
        print(f"Error: The file '{args.input_file}' was not found.")
        sys.exit(1)

    try:
        # 3. Read Input File (Force UTF-8 for Sanskrit support)
        print(f"Reading from {args.input_file}...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        # 4. Perform Conversion
        result_data = convert_sanskrit_md_to_json(markdown_text)

        # 5. Write Output File
        print(f"Writing to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            # ensure_ascii=False is critical for readable Sanskrit in the JSON
            json.dump(result_data, f, indent=4, ensure_ascii=False)

        print("Success! Conversion complete.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()