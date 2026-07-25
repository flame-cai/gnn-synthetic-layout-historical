"""Generate manuscript-level layout and grapheme statistics from PAGE-XML."""

from __future__ import annotations

import html
import importlib.util
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
MANUSCRIPTS = {
    "circle_new": {
        "directory": ROOT / "app/input_manuscripts/circle_new",
        "layout_type": "circular text, complex layout",
    },
    "dense": {
        "directory": ROOT / "app/input_manuscripts/dense",
        "layout_type": "single column, dense marginalia",
    },
    "yajn": {
        "directory": ROOT / "app/input_manuscripts/yajn",
        "layout_type": "single column, moderate marginalia",
    },
}
PAGE_NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def load_tokenizer():
    path = ROOT / "app/input_manuscripts/get_graphemes.py"
    spec = importlib.util.spec_from_file_location("get_graphemes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load grapheme tokenizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_gc_tokens


def summary(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "max": max(values), "average": statistics.mean(values)}


def compact(value: float) -> str:
    return f"{value:.2f}" if not float(value).is_integer() else str(int(value))


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def normalize_mojibake(text: str) -> str:
    """Recover UTF-8 text that was stored as Latin-1-compatible mojibake."""
    try:
        recovered = text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text
    return recovered if any("\u0900" <= char <= "\u097f" for char in recovered) else text


def main() -> None:
    get_gc_tokens = load_tokenizer()
    all_clusters: Counter[str] = Counter()
    manuscripts: dict[str, dict] = {}

    for name, config in MANUSCRIPTS.items():
        xml_dir = config["directory"] / "layout_analysis_output/page-xml-format"
        page_rows = []
        for xml_path in sorted(xml_dir.glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            text_lines = root.findall(".//p:TextLine", PAGE_NS)
            texts = [
                normalize_mojibake(line.findtext("p:TextEquiv/p:Unicode", default="", namespaces=PAGE_NS))
                for line in text_lines
            ]
            clusters = [cluster for text in texts for cluster in get_gc_tokens(text)]
            all_clusters.update(clusters)
            page_rows.append({
                "page_id": xml_path.stem,
                "text_line_count": len(text_lines),
                "grapheme_cluster_count": len(clusters),
            })

        effort = json.loads((config["directory"] / "layout_analysis_output/layout_effort.json").read_text(encoding="utf-8"))
        page_effort = [entry["totals"] for entry in effort["pages"].values()]
        manual_seconds = [entry["active_edit_time_seconds"] for entry in page_effort]
        orientation_counts: Counter[str] = Counter()
        if name == "circle_new":
            for metadata_path in sorted(xml_dir.glob("*_reading_direction_metadata.json")):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                for annotation in metadata.get("line_annotations", []):
                    if annotation.get("status") != "active":
                        continue
                    direction = annotation.get("reading_direction", [])
                    if len(direction) == 2:
                        angle = math.degrees(math.atan2(direction[1], direction[0])) % 360
                        orientation_counts[f"{angle:.0f} degrees"] += 1

        manuscripts[name] = {
            "layout_type": config["layout_type"],
            "page_count": len(page_rows),
            "text_lines_per_page": summary([row["text_line_count"] for row in page_rows]),
            "grapheme_clusters_per_page": summary([row["grapheme_cluster_count"] for row in page_rows]),
            "manual_layout_correction_time_seconds_per_page": summary(manual_seconds),
            "manual_layout_correction_time_minutes_per_page": {
                key: value / 60 for key, value in summary(manual_seconds).items()
            },
            "nodes_added": sum(entry["nodes_added"] for entry in page_effort),
            "nodes_deleted": sum(entry["nodes_deleted"] for entry in page_effort),
            "edges_added": sum(entry["edges_added"] for entry in page_effort),
            "edges_deleted": sum(entry["edges_deleted"] for entry in page_effort),
            "pages": page_rows,
            "circular_text_line_orientations": dict(sorted(orientation_counts.items())),
        }

    top_ten = sorted(all_clusters.items(), key=lambda item: (-item[1], item[0]))[:10]
    least_ten = sorted(all_clusters.items(), key=lambda item: (item[1], item[0]))[:10]
    payload = {
        "source": {
            "page_xml_schema": "PAGE-XML 2013-07-15",
            "grapheme_tokenizer": "app/input_manuscripts/get_graphemes.py:get_gc_tokens",
            "manual_correction_time": "active_edit_time_seconds from latest per-page totals in layout_effort.json",
            "edit_counts": "sum of latest per-page totals in layout_effort.json; revised pages are counted once",
        },
        "manuscripts": manuscripts,
        "combined_grapheme_clusters": {
            "total_cluster_count": sum(all_clusters.values()),
            "unique_cluster_count": len(all_clusters),
            "top_10": [{"cluster": cluster, "frequency": frequency} for cluster, frequency in top_ten],
            "least_10": [{"cluster": cluster, "frequency": frequency} for cluster, frequency in least_ten],
        },
    }
    (OUTPUT / "dataset_statistics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def plot(title: str, values: list[tuple[str, int]], x: int) -> str:
        width, height, baseline, chart_height = 780, 600, 470, 350
        max_value = max(value for _, value in values)
        bar_width, gap = 48, 22
        bars = []
        for index, (label, value) in enumerate(values):
            bar_height = value / max_value * chart_height
            left = 70 + index * (bar_width + gap)
            top = baseline - bar_height
            bars.append(
                f'<rect x="{left}" y="{top:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="#2c6e9f" rx="3"/>'
                f'<text x="{left + bar_width / 2}" y="{top - 8:.1f}" text-anchor="middle" class="value">{value}</text>'
                f'<text x="{left + bar_width / 2}" y="{baseline + 18}" text-anchor="end" transform="rotate(-42 {left + bar_width / 2} {baseline + 18})" class="tick">{html.escape(label)}</text>'
            )
        return f'''<g transform="translate({x},0)">
          <text x="390" y="42" text-anchor="middle" class="title">{title}</text>
          <line x1="70" y1="120" x2="70" y2="470" class="axis"/><line x1="70" y1="470" x2="755" y2="470" class="axis"/>
          <text x="22" y="300" text-anchor="middle" transform="rotate(-90 22 300)" class="axis-label">Frequency</text>
          {''.join(bars)}
        </g>'''

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="600" viewBox="0 0 1600 600">
      <style>text {{ font-family: 'Noto Sans Devanagari', 'Nirmala UI', Arial, sans-serif; fill: #17212b; }} .title {{ font-size: 25px; font-weight: 700; }} .axis-label {{ font-size: 19px; }} .tick {{ font-size: 20px; }} .value {{ font-size: 16px; }} .axis {{ stroke: #52606d; stroke-width: 1.5; }}</style>
      {plot('(a) Ten most common grapheme clusters', top_ten, 0)}
      {plot('(b) Ten least common grapheme clusters', least_ten, 800)}
    </svg>'''
    html_page = f"<!doctype html><html><head><meta charset='utf-8'><style>html,body{{margin:0;background:white}}svg{{display:block}}</style></head><body>{svg}</body></html>"
    (OUTPUT / "grapheme_frequency_figure.html").write_text(html_page, encoding="utf-8")

    rows1 = []
    rows2 = []
    layout_type_tex = {
        "circle_new": r"\shortstack[l]{Circular text;\\complex layout}",
        "dense": r"\shortstack[l]{Single column;\\dense marginalia}",
        "yajn": r"\shortstack[l]{Single column;\\moderate marginalia}",
    }
    for name, data in manuscripts.items():
        tl, gc = data["text_lines_per_page"], data["grapheme_clusters_per_page"]
        rows1.append(f"{tex_escape(name)} & {data['page_count']} & {compact(tl['min'])} & {compact(tl['max'])} & {compact(tl['average'])} & {compact(gc['min'])} & {compact(gc['max'])} & {compact(gc['average'])} \\\\")
        time = data["manual_layout_correction_time_minutes_per_page"]
        rows2.append(f"{tex_escape(name)} & {layout_type_tex[name]} & {compact(time['min'])} & {compact(time['max'])} & {compact(time['average'])} & {data['nodes_added']} & {data['nodes_deleted']} & {data['edges_added']} & {data['edges_deleted']} \\\\")
    orientation_count = sum(manuscripts["circle_new"]["circular_text_line_orientations"].values())
    latex = rf'''% Generated by generate_dataset_stats.py. Compile from this directory.
% Requires: \usepackage{{booktabs,graphicx}}
\begin{{table}}[t]
\centering
\caption{{Dataset text-line and grapheme-cluster statistics from PAGE-XML 2013-07-15.}}
\label{{tab:dataset-content-statistics}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{@{{}}lcrrrrrr@{{}}}}
\toprule
Manuscript & Pages & \multicolumn{{3}}{{c}}{{\shortstack{{Text-lines\\per page}}}} & \multicolumn{{3}}{{c}}{{\shortstack{{Grapheme clusters\\per page}}}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}}
 & & \multicolumn{{1}}{{c}}{{Min.}} & \multicolumn{{1}}{{c}}{{Max.}} & \multicolumn{{1}}{{c}}{{Mean}} & \multicolumn{{1}}{{c}}{{Min.}} & \multicolumn{{1}}{{c}}{{Max.}} & \multicolumn{{1}}{{c}}{{Mean}} \\
\midrule
{chr(10).join(rows1)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table*}}[t]
\centering
\caption{{Manual layout-correction effort. Time is active correction time per page in minutes; edit counts are manuscript totals.}}
\label{{tab:dataset-layout-effort}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.16}}
\begin{{tabular}}{{@{{}}lp{{2.65cm}}rrrrrrrr@{{}}}}
\toprule
Manuscript & Layout type & \multicolumn{{3}}{{c}}{{Correction time/page (min)}} & \multicolumn{{2}}{{c}}{{Node edits}} & \multicolumn{{2}}{{c}}{{Edge edits}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-7}} \cmidrule(lr){{8-9}}
 & & Min. & Max. & Mean & Added & Deleted & Added & Deleted \\
\midrule
{chr(10).join(rows2)}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{grapheme_frequency_figure.png}}
\caption{{Frequency of atomic grapheme clusters across all manuscripts. The \texttt{{circle\_new}} manuscript contains {orientation_count} active manual circular text-line orientation annotations.}}
\label{{fig:grapheme-frequency}}
\end{{figure*}}
'''
    (OUTPUT / "dataset_statistics.tex").write_text(latex, encoding="utf-8")

    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    subprocess.run([
        str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
        "--window-size=1600,600", f"--screenshot={OUTPUT / 'grapheme_frequency_figure.png'}",
        (OUTPUT / "grapheme_frequency_figure.html").as_uri(),
    ], check=True)


if __name__ == "__main__":
    main()
