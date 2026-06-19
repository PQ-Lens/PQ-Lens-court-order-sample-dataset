from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .schemas import TrainingExample


class DatasetLoadError(ValueError):
    """Raised when a configured dataset cannot be loaded."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetLoadError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetLoadError(f"JSONL row must be an object at {path}:{line_no}")
            rows.append(row)
    return rows


def _read_tabular(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    try:
        import pandas as pd
    except ImportError as exc:
        raise DatasetLoadError("XLSX/Parquet loading requires pandas and openpyxl/pyarrow.") from exc

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path).fillna("").to_dict(orient="records")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).fillna("").to_dict(orient="records")
    raise DatasetLoadError(f"Unsupported local dataset extension: {path.suffix}")


def _fetch_api_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_url = str(config["base_url"]).rstrip("/")
    dataset_id = urllib.parse.quote(str(config["dataset_id"]), safe="")
    batch_size = int(config.get("batch_size", 1000))
    cursor = 0
    rows: list[dict[str, Any]] = []

    while True:
        url = f"{base_url}/datasets/{dataset_id}/records?batch_size={batch_size}&cursor={cursor}"
        with urllib.request.urlopen(url, timeout=int(config.get("timeout_seconds", 30))) as response:
            body = json.loads(response.read().decode("utf-8"))
        data = body.get("data", [])
        if not isinstance(data, list):
            raise DatasetLoadError(f"Unexpected API response shape from {url}")
        rows.extend(data)
        next_cursor = body.get("next_cursor")
        if next_cursor is None or not data:
            break
        cursor = int(next_cursor)
    return rows


def _load_hf_dataset(config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise DatasetLoadError("Hugging Face dataset loading requires the datasets package.") from exc

    name = config["name"]
    subset = config.get("subset")
    split = config.get("split", "train")
    dataset = load_dataset(name, subset, split=split) if subset else load_dataset(name, split=split)
    max_rows = config.get("max_rows")
    if max_rows:
        dataset = dataset.select(range(min(int(max_rows), len(dataset))))
    return [dict(row) for row in dataset]


def _nested_get(row: dict[str, Any], path: str | None, default: Any = "") -> Any:
    if not path:
        return default
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    messages = []
    for item in value:
        if isinstance(item, dict) and item.get("content"):
            messages.append({"role": str(item.get("role", "user")), "content": str(item["content"])})
    return messages


def normalize_rows(rows: Iterable[dict[str, Any]], config: dict[str, Any]) -> list[TrainingExample]:
    columns = config.get("columns", {})
    examples = []
    for index, row in enumerate(rows):
        metadata = dict(row.get("metadata") or row.get("attributes") or {})
        provenance = row.get("provenance")
        if isinstance(provenance, dict):
            metadata["provenance"] = provenance

        example_id = str(_nested_get(row, columns.get("id"), row.get("id") or f"row_{index:06d}"))
        prompt = str(_nested_get(row, columns.get("prompt"), row.get("prompt", "")) or "")
        completion = str(_nested_get(row, columns.get("completion"), row.get("completion", "")) or "")
        text = str(_nested_get(row, columns.get("text"), row.get("text", "")) or "")
        messages = _normalize_messages(_nested_get(row, columns.get("messages"), row.get("messages")))
        target_from_metadata = None
        translation_metadata = row.get("translation_metadata")
        if isinstance(translation_metadata, dict):
            target_from_metadata = translation_metadata.get("target_text")
            metadata["translation_metadata"] = translation_metadata

        if target_from_metadata and not completion:
            completion = str(target_from_metadata)

        source_lang = _nested_get(row, columns.get("source_lang"), row.get("language"))
        target_lang = _nested_get(row, columns.get("target_lang"), None)
        if isinstance(translation_metadata, dict):
            source_lang = translation_metadata.get("source_language", source_lang)
            target_lang = translation_metadata.get("target_language", target_lang)

        group_id = _nested_get(row, columns.get("group_id"), None)
        if not group_id and isinstance(provenance, dict):
            group_id = provenance.get("source_url") or provenance.get("source_text")

        examples.append(
            TrainingExample(
                id=example_id,
                prompt=prompt,
                completion=completion,
                text=text,
                messages=messages,
                source_lang=str(source_lang) if source_lang else None,
                target_lang=str(target_lang) if target_lang else None,
                group_id=str(group_id) if group_id else example_id,
                metadata=metadata,
            )
        )
    return examples


def load_dataset_examples(config: dict[str, Any]) -> list[TrainingExample]:
    source = config.get("source", {})
    source_type = source.get("type")
    if source_type == "local":
        path = Path(source["path"]).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        rows = _read_jsonl(path) if path.suffix.lower() == ".jsonl" else _read_tabular(path)
    elif source_type == "api":
        rows = _fetch_api_records(source)
    elif source_type == "huggingface":
        rows = _load_hf_dataset(source)
    else:
        raise DatasetLoadError(f"Unsupported dataset source type: {source_type}")

    examples = normalize_rows(rows, config)
    if not examples:
        raise DatasetLoadError("Dataset loader returned no examples")
    return examples
