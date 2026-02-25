# PQ-Lens-court-order-sample-dataset

This repository now includes a minimal Flask API for serving dataset rows from an Excel file.

## Configuration

Dataset settings are stored in `datasets_config.json`.
Each configured dataset has:

- `name`: dataset identifier used in API calls
- `description`: human-readable description
- `datafile`: file name to load (for example `bilingual.xlsx`)

Example:

```json
{
  "datasets": [
    {
      "name": "court_orders_bilingual",
      "description": "Sample bilingual court-order metadata dataset.",
      "datafile": "bilingual.xlsx"
    }
  ]
}
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
python app.py
```

The service starts on `http://localhost:5000`.

## Endpoints

### `GET /datasets`
Returns all configured datasets.

### `GET /data`
Returns rows from the requested dataset.

#### Required query parameter

- `dataset`: dataset name from `datasets_config.json`

#### Optional query parameters

- `columns`: comma-separated list of columns to return
  - Example: `columns=col_a,col_b`
- `filters`: comma-separated exact-match filters in `column:value` format
  - Example: `filters=status:active,lang:en`
- `page`: page number (1-based)
- `per_page`: number of rows per page

If `page`/`per_page` are provided, paginated output is returned.

## Example requests

Get all rows:

```bash
curl "http://localhost:5000/data?dataset=court_orders_bilingual"
```

Get selected columns:

```bash
curl "http://localhost:5000/data?dataset=court_orders_bilingual&columns=ID,English"
```

Get filtered rows:

```bash
curl "http://localhost:5000/data?dataset=court_orders_bilingual&filters=Language:English"
```

Get paginated rows:

```bash
curl "http://localhost:5000/data?dataset=court_orders_bilingual&page=1&per_page=25"
```
