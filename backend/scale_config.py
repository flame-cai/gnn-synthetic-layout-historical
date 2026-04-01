import json
from pathlib import Path


SCALE_CONFIG_FILENAME = "segmentation_scale_config.json"
ENABLED_DEFAULT_X_SCALE = 0.25
ENABLED_DEFAULT_Y_SCALE = 0.5


def _to_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalize_scale_config(enabled=False, x_scale=1.0, y_scale=1.0):
    if not _to_bool(enabled):
        return {
            "enabled": False,
            "x_scale": 1.0,
            "y_scale": 1.0,
        }

    normalized = {
        "enabled": True,
        "x_scale": _to_float(x_scale, ENABLED_DEFAULT_X_SCALE),
        "y_scale": _to_float(y_scale, ENABLED_DEFAULT_Y_SCALE),
    }

    if normalized["x_scale"] <= 0:
        normalized["x_scale"] = ENABLED_DEFAULT_X_SCALE
    if normalized["y_scale"] <= 0:
        normalized["y_scale"] = ENABLED_DEFAULT_Y_SCALE

    return normalized


def get_scale_config_path(manuscript_path):
    return Path(manuscript_path) / SCALE_CONFIG_FILENAME


def save_scale_config(manuscript_path, enabled=False, x_scale=1.0, y_scale=1.0):
    config = normalize_scale_config(enabled=enabled, x_scale=x_scale, y_scale=y_scale)
    config_path = get_scale_config_path(manuscript_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def load_scale_config(manuscript_path):
    config_path = get_scale_config_path(manuscript_path)
    if not config_path.exists():
        return normalize_scale_config()

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return normalize_scale_config()

    return normalize_scale_config(
        enabled=raw_config.get("enabled", False),
        x_scale=raw_config.get("x_scale", 1.0),
        y_scale=raw_config.get("y_scale", 1.0),
    )


def scale_x(value, scale_config):
    return value * scale_config["x_scale"]


def scale_y(value, scale_config):
    return value * scale_config["y_scale"]


def restore_x(value, scale_config):
    return value / scale_config["x_scale"]


def restore_y(value, scale_config):
    return value / scale_config["y_scale"]
