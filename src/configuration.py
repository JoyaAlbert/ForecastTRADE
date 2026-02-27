from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_external_config(path: str | None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        try:
            if yaml is not None:
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}


def apply_config_overrides(config_cls, config_data: Dict[str, Any]) -> None:
    for key, value in config_data.items():
        if hasattr(config_cls, key):
            setattr(config_cls, key, value)
