# data_creation/utils.py
import logging
import sys
from typing import List
from pathlib import Path

def setup_logging(log_path: Path):
    """Sets up a logger that prints to console and saves to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

def find_page_ids(data_dir: Path) -> List[str]:
    """Finds all unique page identifiers in the raw data directory."""
    page_ids = set()
    for f in data_dir.glob('*_dims.txt'):
        page_id = f.name.replace('_dims.txt', '')
        page_ids.add(page_id)
    
    sorted_ids = sorted(list(page_ids))
    logging.info(f"Found {len(sorted_ids)} pages in '{data_dir}'.")
    return sorted_ids