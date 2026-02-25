from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "datasets_config.json"

app = Flask(__name__)


class DatasetConfigError(Exception):
    """Raised when dataset configuration is missing or invalid."""


def load_dataset_config() -> list[dict[str, str]]:
    if not CONFIG_PATH.exists():
        raise DatasetConfigError(f"Dataset config was not found at {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise DatasetConfigError("Config must contain a non-empty 'datasets' list")

    return datasets


def find_dataset(dataset_name: str) -> dict[str, str] | None:
    datasets = load_dataset_config()
    return next((d for d in datasets if d.get("name") == dataset_name), None)


def parse_csv_param(param_name: str) -> list[str]:
    raw_value = request.args.get(param_name, "")
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_filters() -> dict[str, str]:
    """
    Parses filters from a query-string value like:
    filters=column_a:value_a,column_b:value_b
    """
    parsed: dict[str, str] = {}
    for item in parse_csv_param("filters"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        if key.strip():
            parsed[key.strip()] = value.strip()
    return parsed


@app.get("/datasets")
def list_datasets() -> Any:
    try:
        datasets = load_dataset_config()
    except DatasetConfigError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"datasets": datasets})


@app.get("/data")
def get_data() -> Any:
    dataset_name = request.args.get("dataset")
    if not dataset_name:
        return jsonify({"error": "Missing required query parameter: dataset"}), 400

    try:
        dataset = find_dataset(dataset_name)
    except DatasetConfigError as exc:
        return jsonify({"error": str(exc)}), 500

    if not dataset:
        return jsonify({"error": f"Dataset '{dataset_name}' was not found"}), 404

    datafile = dataset.get("datafile")
    if not datafile:
        return jsonify({"error": f"Dataset '{dataset_name}' has no datafile configured"}), 500

    data_path = BASE_DIR / datafile
    if not data_path.exists():
        return jsonify({"error": f"Datafile '{datafile}' does not exist"}), 404

    try:
        dataframe = pd.read_excel(data_path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to read dataset file: {exc}"}), 500

    selected_columns = parse_csv_param("columns")
    if selected_columns:
        missing_columns = [col for col in selected_columns if col not in dataframe.columns]
        if missing_columns:
            return (
                jsonify(
                    {
                        "error": "Unknown columns in 'columns' parameter",
                        "missing_columns": missing_columns,
                    }
                ),
                400,
            )
        dataframe = dataframe[selected_columns]

    filters = parse_filters()
    if filters:
        for column_name, value in filters.items():
            if column_name not in dataframe.columns:
                return (
                    jsonify(
                        {
                            "error": "Unknown column in 'filters' parameter",
                            "unknown_column": column_name,
                        }
                    ),
                    400,
                )
            dataframe = dataframe[dataframe[column_name].astype(str) == value]

    page = request.args.get("page", type=int)
    per_page = request.args.get("per_page", type=int)

    records = dataframe.fillna("").to_dict(orient="records")
    total = len(records)

    if page is not None or per_page is not None:
        page = page or 1
        per_page = per_page or 10

        if page < 1 or per_page < 1:
            return jsonify({"error": "page and per_page must be positive integers"}), 400

        start = (page - 1) * per_page
        end = start + per_page
        paginated_records = records[start:end]

        return jsonify(
            {
                "dataset": dataset_name,
                "total_records": total,
                "page": page,
                "per_page": per_page,
                "returned_records": len(paginated_records),
                "data": paginated_records,
            }
        )

    return jsonify({"dataset": dataset_name, "total_records": total, "data": records})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
