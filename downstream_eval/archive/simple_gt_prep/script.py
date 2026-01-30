#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update PAGE-XML main text-line content based on digitized ground truth text
stored in a markdown file.

Key Logic:
----------
1. Markdown file contains sections: "Page No. X".
2. Extract text lines per page.
3. Each PAGE-XML file has multiple TextRegions; the largest one (by polygon area)
   corresponds to the main text block (excluding marginalia, page numbers).
4. Replace TextEquiv/Unicode text for each TextLine in that region.
5. Collect and report page mismatches: differing line counts between markdown and XML.

This script contains:
- Strong logging
- Assert statements for safety
- Clean end-of-run mismatch report
"""

import os
import re
import logging
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Global dictionary to collect line-count mismatches
MISMATCH_PAGES = {}   # {page_no: (markdown_count, xml_count)}


# -----------------------------------------------------------------------------
# STEP 1: Parse Markdown File into Page → Lines Mapping
# -----------------------------------------------------------------------------
def parse_markdown_pages(md_path):
    logging.info(f"Parsing markdown file: {md_path}")
    assert os.path.exists(md_path), f"Markdown file not found: {md_path}"

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Capture "Page No. X" and everything after it, until next page block
    page_pattern = re.compile(r"Page No\. *(\d+)\s*(.*?)\s*(?=Page No\.|\Z)", re.S)

    pages = {}
    for page_no, page_text in page_pattern.findall(text):
        lines = [
            ln.strip()
            for ln in page_text.splitlines()
            if ln.strip()
        ]
        pages[int(page_no)] = lines
        logging.info(f"Page {page_no}: extracted {len(lines)} markdown lines.")

    assert len(pages) > 0, "No pages detected in markdown file!"

    return pages


# -----------------------------------------------------------------------------
# STEP 2: Compute polygon area from PAGE-XML Coords
# -----------------------------------------------------------------------------
def polygon_area_from_coords(coords_str):
    """Compute polygon area from PAGE coords 'x1,y1 x2,y2 ...' using shoelace formula."""
    pts = []
    for p in coords_str.split():
        x, y = p.split(',')
        pts.append((float(x), float(y)))

    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


# -----------------------------------------------------------------------------
# STEP 3: Select the Main Text Region (Largest TextRegion)
# -----------------------------------------------------------------------------
def get_main_text_region(page_root):
    ns = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

    regions = page_root.findall(".//p:TextRegion", ns)
    assert regions, "No TextRegion found in PAGE-XML!"

    region_areas = []
    for r in regions:
        coords = r.find("p:Coords", ns)
        assert coords is not None, "TextRegion missing <Coords>"
        area = polygon_area_from_coords(coords.attrib["points"])
        region_areas.append((area, r))

    region_areas.sort(reverse=True, key=lambda x: x[0])
    biggest_region = region_areas[0][1]

    logging.info(f"Selected main region with area {region_areas[0][0]:.1f}")
    return biggest_region


# -----------------------------------------------------------------------------
# STEP 4: Update TextLines inside the Main TextRegion
# -----------------------------------------------------------------------------
def update_page_xml(xml_path, page_lines, page_no):
    logging.info(f"\nUpdating PAGE-XML: {xml_path} (page {page_no})")

    assert os.path.exists(xml_path), f"PAGE-XML not found: {xml_path}"

    ns = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    ET.register_namespace("", ns["p"])  # ensure namespace preserved

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- Identify main text region
    main_region = get_main_text_region(root)

    # --- Extract text lines from region
    textlines = main_region.findall("p:TextLine", ns)
    xml_count = len(textlines)
    md_count = len(page_lines)

    logging.info(f"XML text lines in main region: {xml_count}")
    logging.info(f"Markdown lines for this page: {md_count}")

    # --- Record mismatch if counts differ
    if md_count != xml_count:
        logging.warning(f"Line count mismatch for page {page_no}: MD={md_count} XML={xml_count}")
        MISMATCH_PAGES[page_no] = (md_count, xml_count)

    # Replace only up to min length
    min_len = min(md_count, xml_count)

    # --- Update each line
    for i in range(min_len):
        xml_line = textlines[i]
        md_line = page_lines[i]

        te = xml_line.find("p:TextEquiv", ns)
        if te is None:
            te = ET.SubElement(xml_line, "TextEquiv")

        unicode_el = te.find("p:Unicode", ns)
        if unicode_el is None:
            unicode_el = ET.SubElement(te, "Unicode")

        unicode_el.text = md_line
        logging.info(f"  Updated line {i+1}/{min_len}: {md_line[:60]}")

    # Save updated XML
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    logging.info(f"Saved updated PAGE-XML → {xml_path}")


# -----------------------------------------------------------------------------
# STEP 5: Main Coordinator — Iterate All Pages and XML Files
# -----------------------------------------------------------------------------
def update_all_pages(md_path, xml_folder):
    pages = parse_markdown_pages(md_path)

    for page_no, lines in pages.items():
        xml_filename = f"3976_{page_no:04d}.xml"
        xml_path = os.path.join(xml_folder, xml_filename)

        if not os.path.exists(xml_path):
            logging.error(f"PAGE-XML missing for page {page_no}: {xml_path}")
            continue

        update_page_xml(xml_path, lines, page_no)

    # -------------------------------------------------------------------------
    # PRINT END-OF-RUN MISMATCH SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: LINE COUNT MISMATCHES (Markdown vs PAGE-XML)")
    print("=" * 70)

    if not MISMATCH_PAGES:
        print("✔ No mismatches — all pages aligned perfectly!")
    else:
        for pg, (md, xml) in sorted(MISMATCH_PAGES.items()):
            print(f"• Page {pg}: MD={md}  XML={xml}")

    print("=" * 70 + "\n")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update PAGE-XML from markdown ground truth text.")
    parser.add_argument("--md", required=True, help="Path to markdown file (vadakautuhala.md)")
    parser.add_argument("--xml_folder", required=True, help="Folder containing PAGE-XML files")

    args = parser.parse_args()
    update_all_pages(args.md, args.xml_folder)
