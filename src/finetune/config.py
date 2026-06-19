from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigError("YAML configs require PyYAML. Use JSON configs or install PyYAML.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ConfigError(f"Config must be an object: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    experiment_path = Path(path).resolve()
    root = experiment_path.parents[1] if experiment_path.parent.name == "experiments" else experiment_path.parent
    experiment = _load_mapping(experiment_path)

    resolved: dict[str, Any] = {}
    for key, directory in [
        ("model_config", "models"),
        ("dataset_config", "datasets"),
        ("method_config", "methods"),
        ("evaluation_config", "evaluation"),
    ]:
        ref = experiment.pop(key, None)
        if not ref:
            continue
        ref_path = Path(ref)
        if not ref_path.is_absolute():
            ref_path = root / directory / ref_path
        resolved = deep_merge(resolved, _load_mapping(ref_path))

    return deep_merge(resolved, experiment)


def require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required config section: {section}")
    return value
