# PQ-Lens Court Order Sample Dataset API

A lightweight Flask API for managing **dataset metadata** and **text records** for court-order related NLP workflows.

This project is designed as a small, local-first service that stores data in a JSON file and exposes endpoints for:

- dataset CRUD
- record CRUD
- paginated record retrieval
- deterministic shuffled sampling
- split generation for evaluation/dev/test workflows

---

## Table of contents

- [Architecture at a glance](#architecture-at-a-glance)
- [Data model](#data-model)
  - [Dataset object](#dataset-object)
  - [Record object](#record-object)
  - [Provenance object](#provenance-object)
- [Validation rules](#validation-rules)
- [Storage model](#storage-model)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API reference](#api-reference)
  - [Health and metadata](#health-and-metadata)
  - [Dataset endpoints](#dataset-endpoints)
  - [Record endpoints](#record-endpoints)
  - [Split sampling endpoint](#split-sampling-endpoint)
- [Examples](#examples)
- [Error handling](#error-handling)
- [Testing](#testing)
- [Repository notes](#repository-notes)

---

## Architecture at a glance

The application is implemented in `app.py` and built around:

1. **Flask app factory (`create_app`)**
2. **`DataStore` abstraction** for JSON-file persistence
3. **Validation helpers** for datasets, records, and query/body parameters
4. **REST endpoints** under `/datasets` and nested record routes

Data is persisted to `data_store.json` by default (created automatically at runtime if absent).

---

## Dataset scope and curation

This repository contains a **curated sample corpus of publicly available legal/court-order data** collected from multiple public-domain or publicly accessible sources.

Key characteristics of the corpus:

- **Cross-source curation**: documents are aggregated from different public sources rather than a single publisher.
- **Language distribution**: the dataset is weighted toward **Maltese-English paired content** and **Maltese-only content**, while also including **English-only records**.
- **Translation relevance**: because many entries are bilingual or translation-adjacent, the corpus is well suited for building and evaluating **Maltese ↔ English translation models**.

Recommended use cases:

- machine translation (MT) fine-tuning for Maltese-English
- bilingual alignment experiments
- terminology extraction and domain adaptation for legal text

> Note: suitability for production ML depends on your own quality checks (deduplication, licensing review, alignment verification, and train/dev/test protocol design).

---

## Data model

### Dataset object

A dataset represents a logical collection of records.

```json
{
  "id": "court_orders",
  "summary": "Court order metadata.",
  "description": "Optional long description",
  "language_mode": "bilingual_aligned",
  "synthetic_status": "non_synthetic",
  "synthetic_details": {},
  "provenance": {
    "source_url": "https://example.com/public-registry",
    "source_type": "web"
  },
  "created_at": "2026-01-01T10:00:00+00:00",
  "updated_at": "2026-01-01T10:00:00+00:00"
}
```

### Record object

A record belongs to exactly one dataset.

```json
{
  "id": "r1",
  "dataset_id": "court_orders",
  "text": "Sample text",
  "language": "eng_Latn",
  "language_pair": null,
  "synthetic_status": "unknown",
  "translation_metadata": {},
  "provenance": {
    "source_text": "OCR process from archive",
    "source_type": "ocr_archive"
  },
  "attributes": {},
  "created_at": "2026-01-01T10:00:00+00:00",
  "updated_at": "2026-01-01T10:00:00+00:00"
}
```

### Provenance object

Provenance is mandatory for datasets and records and must include at least one of:

- `source_url`
- `source_text`

Optional:

- `source_type` (enumerated values below)

---

## Validation rules

### Enumerated values

- `language_mode`:
  - `monolingual`
  - `bilingual_aligned`
  - `multilingual`

- `synthetic_status`:
  - `non_synthetic`
  - `partly_synthetic`
  - `fully_synthetic`
  - `unknown`

- `source_type`:
  - `web`
  - `api`
  - `file`
  - `ocr_archive`
  - `manual_entry`
  - `other`

### Language format

Record `language` must match NLLB tag format like `eng_Latn` or `mlt_Latn`.

### OCR-specific rule

When `provenance.source_type == "ocr_archive"`, `provenance.source_text` is required.

### Query parameter checks

- `batch_size`: positive integer (`1..10000`)
- `cursor`: non-negative integer (`0..10000000`)
- `seed`: optional integer
- `order`: `natural` or `shuffle`

---

## Storage model

The API stores state in a JSON file with this shape:

```json
{
  "datasets": {
    "<dataset_id>": { "...dataset object...": "..." }
  },
  "records": {
    "<dataset_id>": {
      "<record_id>": { "...record object...": "..." }
    }
  }
}
```

Persistence uses atomic write semantics via temporary file replacement.

---

## Prerequisites

- Python 3.10+
- pip

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the API

```bash
python app.py
```

Server defaults:

- Host: `0.0.0.0`
- Port: `5000`
- Debug: enabled when run via `python app.py`

Base URL (local): `http://localhost:5000`

---

## API reference

## Health and metadata

### `GET /health`
Returns service liveness.

### `GET /languages`
Returns currently advertised language codes.

---

## Dataset endpoints

### `GET /datasets`
List all datasets.

### `POST /datasets`
Create a dataset.

Required body fields:

- `id`
- `summary`
- `language_mode`
- `synthetic_status`
- `provenance`

### `GET /datasets/<dataset_id>`
Get one dataset.

### `PATCH /datasets/<dataset_id>`
Partially update dataset fields.

Protected/immutable fields:

- `id`
- `created_at`

### `DELETE /datasets/<dataset_id>`
Delete dataset and all associated records.

---

## Record endpoints

### `POST /datasets/<dataset_id>/records`
Create a record inside a dataset.

Required body fields:

- `id`
- `text`
- `language`
- `provenance`

### `GET /datasets/<dataset_id>/records`
List records with optional filtering, ordering, and cursor pagination.

Query params:

- `order`: `natural` (default) or `shuffle`
- `seed`: integer (used only with `order=shuffle` for reproducibility)
- `batch_size`: integer, default `100`
- `cursor`: integer offset, default `0`
- `language`: exact language filter
- `synthetic_status`: exact synthetic status filter

Response includes:

- `total_records`
- `batch_size`
- `cursor`
- `next_cursor` (or `null` when exhausted)
- `order`
- `seed`
- `data` (current batch)

### `GET /datasets/<dataset_id>/records/<record_id>`
Fetch one record.

### `PATCH /datasets/<dataset_id>/records/<record_id>`
Partially update one record.

Protected/immutable fields:

- `id`
- `dataset_id`
- `created_at`

### `DELETE /datasets/<dataset_id>/records/<record_id>`
Delete one record.

---

## Split sampling endpoint

### `POST /datasets/<dataset_id>/splits/sample`
Build named split batches from dataset records.

Request body:

- `splits` (required): object mapping split names to sizes
  - example: `{ "dev": 100, "test": 100, "eval": 50 }`
- `order` (optional): `natural` or `shuffle` (default: `shuffle`)
- `seed` (optional): integer seed for deterministic shuffle
- `disjoint` (optional): boolean (default: `true`)
- `filters` (optional):
  - `language`
  - `synthetic_status`

Behavior:

- If `disjoint=true`, records are consumed split-by-split without overlap.
- If `disjoint=false`, each split is taken independently from the start of the same filtered list.

---

## Examples

### 1) Create dataset

```bash
curl -X POST "http://localhost:5000/datasets" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "court_orders",
    "summary": "Court order metadata",
    "description": "Bilingual corpus for extraction tasks",
    "language_mode": "bilingual_aligned",
    "synthetic_status": "non_synthetic",
    "provenance": {
      "source_url": "https://example.com/public-registry",
      "source_type": "web"
    }
  }'
```

### 2) Create record

```bash
curl -X POST "http://localhost:5000/datasets/court_orders/records" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "r1",
    "text": "This is a sample court order record.",
    "language": "eng_Latn",
    "provenance": {
      "source_text": "Manual extraction from registry",
      "source_type": "manual_entry"
    }
  }'
```

### 3) List records with deterministic shuffling

```bash
curl "http://localhost:5000/datasets/court_orders/records?order=shuffle&seed=42&batch_size=25"
```

### 4) Sample disjoint splits

```bash
curl -X POST "http://localhost:5000/datasets/court_orders/splits/sample" \
  -H "Content-Type: application/json" \
  -d '{
    "splits": {"dev": 3, "test": 3, "eval": 2},
    "order": "shuffle",
    "seed": 12,
    "disjoint": true
  }'
```

---

## Error handling

Errors are returned as JSON with a predictable structure:

```json
{
  "error": "human-readable message",
  "details": {},
  "timestamp": "2026-01-01T10:00:00+00:00"
}
```

Common status codes:

- `400` invalid input
- `404` not found
- `409` duplicate dataset/record IDs
- `500` unexpected server error

---

## Testing

Run the test suite:

```bash
python -m unittest -v
```

Current tests validate:

- provenance requirements
- NLLB language validation
- natural ordering defaults
- shuffle reproducibility with seed
- disjoint split behavior

---

## Repository notes

- PDF and other source artifacts in this repository represent a curated, publicly available legal-text corpus with strong Maltese-English relevance; however, the Flask API runtime persistence path is JSON-based (`data_store.json`) and not Excel-backed.
- To use another storage location (for tests/environments), pass `DATA_STORE_PATH` via `create_app({...})`.
