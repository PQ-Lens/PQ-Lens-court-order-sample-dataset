from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STORE_PATH = BASE_DIR / "data_store.json"
NLLB_TAG_PATTERN = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")

LANGUAGE_MODES = {"monolingual", "bilingual_aligned", "multilingual"}
SYNTHETIC_STATUS = {"non_synthetic", "partly_synthetic", "fully_synthetic", "unknown"}
SOURCE_TYPES = {"web", "api", "file", "ocr_archive", "manual_entry", "other"}
ORDERS = {"natural", "shuffle"}


class ApiError(Exception):
    def __init__(self, message: str, status_code: int = 400, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _initial_state(self) -> dict[str, Any]:
        return {"datasets": {}, "records": {}}

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            state = self._initial_state()
            self.save(state)
            return state

        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, state: dict[str, Any]) -> None:
        temp_path = self.path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        temp_path.replace(self.path)


def parse_positive_int(name: str, default: int, *, minimum: int = 1, maximum: int = 10_000) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover
        raise ApiError(f"{name} must be an integer", 400) from exc

    if value < minimum or value > maximum:
        raise ApiError(f"{name} must be between {minimum} and {maximum}", 400)
    return value


def parse_optional_int(name: str) -> int | None:
    raw = request.args.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ApiError(f"{name} must be an integer", 400) from exc


def validate_provenance(payload: dict[str, Any], *, require_source_text_for_ocr: bool = True) -> None:
    source_url = payload.get("source_url")
    source_text = payload.get("source_text")
    source_type = payload.get("source_type")

    if not source_url and not source_text:
        raise ApiError("provenance must include source_url or source_text", 400)

    if source_type and source_type not in SOURCE_TYPES:
        raise ApiError(f"Invalid provenance.source_type: {source_type}", 400)

    if require_source_text_for_ocr and source_type == "ocr_archive" and not source_text:
        raise ApiError("provenance.source_text is required when source_type is ocr_archive", 400)


def validate_dataset_payload(payload: dict[str, Any], *, partial: bool = False) -> None:
    required = [] if partial else ["id", "summary", "language_mode", "synthetic_status", "provenance"]
    for key in required:
        if key not in payload:
            raise ApiError(f"Missing required field: {key}", 400)

    if "language_mode" in payload and payload["language_mode"] not in LANGUAGE_MODES:
        raise ApiError("Invalid language_mode", 400, {"allowed": sorted(LANGUAGE_MODES)})

    if "synthetic_status" in payload and payload["synthetic_status"] not in SYNTHETIC_STATUS:
        raise ApiError("Invalid synthetic_status", 400, {"allowed": sorted(SYNTHETIC_STATUS)})

    provenance = payload.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ApiError("provenance must be an object", 400)
        validate_provenance(provenance)


def validate_record_payload(payload: dict[str, Any], *, partial: bool = False) -> None:
    required = [] if partial else ["id", "text", "language", "provenance"]
    for key in required:
        if key not in payload:
            raise ApiError(f"Missing required field: {key}", 400)

    if "language" in payload:
        language = payload["language"]
        if not isinstance(language, str) or not NLLB_TAG_PATTERN.match(language):
            raise ApiError("language must use NLLB format such as 'eng_Latn'", 400)

    if "synthetic_status" in payload and payload["synthetic_status"] not in SYNTHETIC_STATUS:
        raise ApiError("Invalid synthetic_status", 400, {"allowed": sorted(SYNTHETIC_STATUS)})

    provenance = payload.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ApiError("provenance must be an object", 400)
        validate_provenance(provenance)


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update({"DATA_STORE_PATH": DEFAULT_STORE_PATH})
    if config:
        app.config.update(config)

    store = DataStore(Path(app.config["DATA_STORE_PATH"]))

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        response = {"error": exc.message, "details": exc.details, "timestamp": utc_now()}
        return jsonify(response), exc.status_code

    @app.errorhandler(404)
    def not_found(_: Any):
        return jsonify({"error": "Resource not found", "timestamp": utc_now()}), 404

    @app.errorhandler(Exception)
    def internal_error(exc: Exception):
        return (
            jsonify({"error": "Internal server error", "details": {"reason": str(exc)}, "timestamp": utc_now()}),
            500,
        )

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok", "timestamp": utc_now()})

    @app.get("/languages")
    def languages() -> Any:
        return jsonify(
            {
                "languages": [
                    {"code": "eng_Latn", "label": "English"},
                    {"code": "mlt_Latn", "label": "Maltese"},
                ]
            }
        )

    @app.get("/datasets")
    def list_datasets() -> Any:
        state = store.load()
        datasets = list(state["datasets"].values())
        return jsonify({"datasets": datasets, "count": len(datasets)})

    @app.post("/datasets")
    def create_dataset() -> Any:
        payload = request.get_json(silent=True) or {}
        validate_dataset_payload(payload)

        state = store.load()
        dataset_id = payload["id"]
        if dataset_id in state["datasets"]:
            raise ApiError(f"Dataset '{dataset_id}' already exists", 409)

        dataset = {
            "id": dataset_id,
            "summary": payload["summary"],
            "description": payload.get("description", ""),
            "language_mode": payload["language_mode"],
            "synthetic_status": payload["synthetic_status"],
            "synthetic_details": payload.get("synthetic_details", {}),
            "provenance": payload["provenance"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

        state["datasets"][dataset_id] = dataset
        state["records"].setdefault(dataset_id, {})
        store.save(state)
        return jsonify({"dataset": dataset}), 201

    @app.get("/datasets/<dataset_id>")
    def get_dataset(dataset_id: str) -> Any:
        state = store.load()
        dataset = state["datasets"].get(dataset_id)
        if not dataset:
            raise ApiError(f"Dataset '{dataset_id}' not found", 404)
        return jsonify({"dataset": dataset})

    @app.patch("/datasets/<dataset_id>")
    def update_dataset(dataset_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        validate_dataset_payload(payload, partial=True)

        state = store.load()
        dataset = state["datasets"].get(dataset_id)
        if not dataset:
            raise ApiError(f"Dataset '{dataset_id}' not found", 404)

        protected = {"id", "created_at"}
        for key, value in payload.items():
            if key in protected:
                continue
            dataset[key] = value
        dataset["updated_at"] = utc_now()
        store.save(state)
        return jsonify({"dataset": dataset})

    @app.delete("/datasets/<dataset_id>")
    def delete_dataset(dataset_id: str) -> Any:
        state = store.load()
        if dataset_id not in state["datasets"]:
            raise ApiError(f"Dataset '{dataset_id}' not found", 404)
        del state["datasets"][dataset_id]
        state["records"].pop(dataset_id, None)
        store.save(state)
        return jsonify({"deleted": True, "dataset_id": dataset_id})

    def get_dataset_or_404(dataset_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        state = store.load()
        dataset = state["datasets"].get(dataset_id)
        if not dataset:
            raise ApiError(f"Dataset '{dataset_id}' not found", 404)
        records_map = state["records"].setdefault(dataset_id, {})
        return state, records_map

    @app.post("/datasets/<dataset_id>/records")
    def create_record(dataset_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        validate_record_payload(payload)

        state, records_map = get_dataset_or_404(dataset_id)
        record_id = payload["id"]
        if record_id in records_map:
            raise ApiError(f"Record '{record_id}' already exists", 409)

        record = {
            "id": record_id,
            "dataset_id": dataset_id,
            "text": payload["text"],
            "language": payload["language"],
            "language_pair": payload.get("language_pair"),
            "synthetic_status": payload.get("synthetic_status", "unknown"),
            "translation_metadata": payload.get("translation_metadata", {}),
            "provenance": payload["provenance"],
            "attributes": payload.get("attributes", {}),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        records_map[record_id] = record
        store.save(state)
        return jsonify({"record": record}), 201

    @app.get("/datasets/<dataset_id>/records")
    def list_records(dataset_id: str) -> Any:
        _, records_map = get_dataset_or_404(dataset_id)

        order = request.args.get("order", "natural")
        if order not in ORDERS:
            raise ApiError("order must be one of natural|shuffle", 400)

        seed = parse_optional_int("seed")
        batch_size = parse_positive_int("batch_size", 100)
        cursor = parse_positive_int("cursor", 0, minimum=0, maximum=10_000_000)

        language_filter = request.args.get("language")
        synthetic_filter = request.args.get("synthetic_status")

        records = list(records_map.values())
        if language_filter:
            records = [row for row in records if row.get("language") == language_filter]
        if synthetic_filter:
            records = [row for row in records if row.get("synthetic_status") == synthetic_filter]

        if order == "shuffle":
            if seed is not None:
                rng = random.Random(seed)
                rng.shuffle(records)
            else:
                random.shuffle(records)

        total = len(records)
        end = cursor + batch_size
        batch = records[cursor:end]
        next_cursor = end if end < total else None

        return jsonify(
            {
                "dataset_id": dataset_id,
                "total_records": total,
                "batch_size": batch_size,
                "cursor": cursor,
                "next_cursor": next_cursor,
                "order": order,
                "seed": seed,
                "data": batch,
            }
        )

    @app.get("/datasets/<dataset_id>/records/<record_id>")
    def get_record(dataset_id: str, record_id: str) -> Any:
        _, records_map = get_dataset_or_404(dataset_id)
        record = records_map.get(record_id)
        if not record:
            raise ApiError(f"Record '{record_id}' not found", 404)
        return jsonify({"record": record})

    @app.patch("/datasets/<dataset_id>/records/<record_id>")
    def update_record(dataset_id: str, record_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        validate_record_payload(payload, partial=True)

        state, records_map = get_dataset_or_404(dataset_id)
        record = records_map.get(record_id)
        if not record:
            raise ApiError(f"Record '{record_id}' not found", 404)

        protected = {"id", "dataset_id", "created_at"}
        for key, value in payload.items():
            if key in protected:
                continue
            record[key] = value
        record["updated_at"] = utc_now()
        store.save(state)
        return jsonify({"record": record})

    @app.delete("/datasets/<dataset_id>/records/<record_id>")
    def delete_record(dataset_id: str, record_id: str) -> Any:
        state, records_map = get_dataset_or_404(dataset_id)
        if record_id not in records_map:
            raise ApiError(f"Record '{record_id}' not found", 404)
        del records_map[record_id]
        store.save(state)
        return jsonify({"deleted": True, "record_id": record_id})

    @app.post("/datasets/<dataset_id>/splits/sample")
    def sample_splits(dataset_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        splits = payload.get("splits")
        if not isinstance(splits, dict) or not splits:
            raise ApiError("splits must be a non-empty object", 400)

        order = payload.get("order", "shuffle")
        if order not in ORDERS:
            raise ApiError("order must be one of natural|shuffle", 400)

        seed = payload.get("seed")
        disjoint = payload.get("disjoint", True)
        filters = payload.get("filters", {})

        _, records_map = get_dataset_or_404(dataset_id)
        records = list(records_map.values())
        if language := filters.get("language"):
            records = [item for item in records if item.get("language") == language]
        if synthetic_status := filters.get("synthetic_status"):
            records = [item for item in records if item.get("synthetic_status") == synthetic_status]

        if order == "shuffle":
            rng = random.Random(seed)
            rng.shuffle(records)

        cursor = 0
        response_splits: dict[str, list[dict[str, Any]]] = {}
        for split_name, split_size in splits.items():
            if not isinstance(split_size, int) or split_size < 0:
                raise ApiError(f"split '{split_name}' must be a non-negative integer", 400)

            if disjoint:
                split_data = records[cursor : cursor + split_size]
                cursor += split_size
            else:
                split_data = records[:split_size]
            response_splits[split_name] = split_data

        return jsonify(
            {
                "dataset_id": dataset_id,
                "order": order,
                "seed": seed,
                "disjoint": disjoint,
                "splits": response_splits,
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
