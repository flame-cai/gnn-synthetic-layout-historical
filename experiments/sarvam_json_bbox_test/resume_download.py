"""Download the output from an already-started Sarvam experiment job."""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
JOB_ID = "20260725_86a96cb2-4213-40f6-91c9-e6f0f8fdf9b1"


def value(obj: object, name: str) -> object:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name)


def main() -> int:
    load_dotenv(ROOT / "app" / ".env")
    client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
    status = client.document_intelligence.get_status(JOB_ID)
    if str(status.job_state).lower() != "completed":
        raise RuntimeError(f"Job is not complete: {status.job_state}")
    downloads = client.document_intelligence.get_download_links(JOB_ID)
    download_urls = value(downloads, "download_urls")
    if not download_urls:
        raise RuntimeError("Completed job did not return a download URL.")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    extracted = OUTPUT_DIR / "extracted"
    extracted.mkdir(parents=True)
    output_name, output = next(iter(download_urls.items()))
    url = value(output, "file_url")
    zip_path = OUTPUT_DIR / str(output_name)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted)
    json_paths = sorted(extracted.rglob("*.json"))
    manifest = {
        "job_id": JOB_ID,
        "job_state": str(status.job_state),
        "requested_output_format": "json",
        "actual_output_format": "html",
        "json_mode_initialise_result": "rejected_by_live_api_http_400",
        "json_files": [str(path.relative_to(OUTPUT_DIR)) for path in json_paths],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("json_files=" + json.dumps(manifest["json_files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
