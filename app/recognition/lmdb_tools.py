from __future__ import annotations

from pathlib import Path

import cv2
import lmdb
import numpy as np


def _check_image_is_valid(image_bytes):
    if image_bytes is None:
        return False
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    image_height, image_width = image.shape[:2]
    return image_height * image_width > 0


def _write_cache(env, cache):
    with env.begin(write=True) as transaction:
        for key, value in cache.items():
            transaction.put(key, value)


def create_lmdb_dataset(input_path: str | Path, gt_file: str | Path, output_path: str | Path, check_valid=True):
    input_path = Path(input_path)
    gt_file = Path(gt_file)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    lines = gt_file.read_text(encoding="utf-8").splitlines()
    estimated_total_bytes = 0
    for raw_line in lines:
        if not raw_line.strip():
            continue
        image_rel_path, _ = raw_line.split("\t", 1)
        image_path = input_path / image_rel_path
        if image_path.exists():
            estimated_total_bytes += image_path.stat().st_size

    map_size = max(int(estimated_total_bytes * 10), 64 * 1024 * 1024)
    env = lmdb.open(str(output_path), map_size=map_size)
    cache = {}
    sample_count = 1

    for raw_line in lines:
        if not raw_line.strip():
            continue

        image_rel_path, label = raw_line.split("\t", 1)
        image_path = input_path / image_rel_path
        if not image_path.exists():
            continue

        image_bytes = image_path.read_bytes()
        if check_valid and not _check_image_is_valid(image_bytes):
            continue

        image_key = f"image-{sample_count:09d}".encode()
        label_key = f"label-{sample_count:09d}".encode()
        cache[image_key] = image_bytes
        cache[label_key] = label.encode("utf-8")

        if sample_count % 1000 == 0:
            _write_cache(env, cache)
            cache = {}
        sample_count += 1

    cache[b"num-samples"] = str(sample_count - 1).encode("utf-8")
    _write_cache(env, cache)
    env.sync()
    env.close()
    return output_path
