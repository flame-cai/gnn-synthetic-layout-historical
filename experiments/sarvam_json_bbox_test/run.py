"""Run one isolated Sarvam Document Digitization JSON/HTML fallback experiment.

This does not modify the application or its existing HTML integration.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI


ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "app" / "input_manuscripts" / "dense" / "images" / "17.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> int:
    load_dotenv(ROOT / "app" / ".env")
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set.")
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    client = SarvamAI(api_subscription_key=api_key)
    # The live endpoint currently rejects ``json`` (HTTP 400) even though the
    # docs advertise it. HTML jobs are documented to bundle the page JSON too.
    output_format = "html"
    job = client.document_intelligence.create_job(language="sa-IN", output_format=output_format)
    print(f"job_id={job.job_id}")
    job.upload_file(str(IMAGE_PATH))
    job.start()
    status = job.wait_until_complete()
    job_state = str(status.job_state)
    print(f"job_state={job_state}")
    if job_state.lower().rsplit(".", 1)[-1] not in {"completed", "complete", "succeeded", "success"}:
        raise RuntimeError(f"Sarvam job did not complete successfully: {job_state}")

    zip_path = OUTPUT_DIR / "sarvam_output.zip"
    job.download_output(str(zip_path))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(OUTPUT_DIR / "extracted")

    json_paths = sorted((OUTPUT_DIR / "extracted").rglob("*.json"))
    manifest = {
        "job_id": str(job.job_id),
        "job_state": job_state,
        "source_image": str(IMAGE_PATH),
        "requested_output_format": "json",
        "actual_output_format": output_format,
        "json_mode_initialise_result": "rejected_by_live_api_http_400",
        "json_files": [str(path.relative_to(OUTPUT_DIR)) for path in json_paths],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("json_files=" + json.dumps(manifest["json_files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
