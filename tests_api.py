import tempfile
import unittest
from pathlib import Path

from app import create_app


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "store.json"
        app = create_app({"TESTING": True, "DATA_STORE_PATH": self.store_path})
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_dataset(self) -> dict:
        response = self.client.post(
            "/datasets",
            json={
                "id": "court_orders",
                "summary": "Court order metadata.",
                "description": "A few sentences describing the dataset.",
                "language_mode": "bilingual_aligned",
                "synthetic_status": "non_synthetic",
                "provenance": {
                    "source_url": "https://example.com/public-registry",
                    "source_type": "web",
                },
            },
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()["dataset"]

    def test_create_dataset_requires_source_url_or_text(self) -> None:
        response = self.client.post(
            "/datasets",
            json={
                "id": "bad_dataset",
                "summary": "Missing provenance source details.",
                "language_mode": "monolingual",
                "synthetic_status": "unknown",
                "provenance": {"source_type": "other"},
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("source_url", response.get_json()["error"])

    def test_create_record_validates_nllb_language_tag(self) -> None:
        self._create_dataset()
        response = self.client.post(
            "/datasets/court_orders/records",
            json={
                "id": "r1",
                "text": "hello",
                "language": "en",
                "provenance": {
                    "source_text": "OCR process from public archive",
                    "source_type": "ocr_archive",
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("NLLB", response.get_json()["error"])

    def test_batch_defaults_to_natural_order(self) -> None:
        self._create_dataset()
        for idx in range(1, 4):
            payload = {
                "id": f"r{idx}",
                "text": f"text-{idx}",
                "language": "eng_Latn",
                "provenance": {
                    "source_text": "Manual registry export",
                    "source_type": "manual_entry",
                },
            }
            response = self.client.post("/datasets/court_orders/records", json=payload)
            self.assertEqual(response.status_code, 201)

        response = self.client.get("/datasets/court_orders/records")
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["order"], "natural")
        self.assertEqual(body["batch_size"], 100)
        self.assertEqual([item["id"] for item in body["data"]], ["r1", "r2", "r3"])

    def test_batch_shuffle_is_reproducible_with_seed(self) -> None:
        self._create_dataset()
        for idx in range(1, 6):
            self.client.post(
                "/datasets/court_orders/records",
                json={
                    "id": f"r{idx}",
                    "text": f"text-{idx}",
                    "language": "eng_Latn",
                    "provenance": {
                        "source_text": "Manual registry export",
                        "source_type": "manual_entry",
                    },
                },
            )

        first = self.client.get(
            "/datasets/court_orders/records?order=shuffle&seed=42&batch_size=5"
        ).get_json()
        second = self.client.get(
            "/datasets/court_orders/records?order=shuffle&seed=42&batch_size=5"
        ).get_json()

        self.assertEqual(
            [item["id"] for item in first["data"]],
            [item["id"] for item in second["data"]],
        )

    def test_split_sampling_returns_disjoint_sets(self) -> None:
        self._create_dataset()
        for idx in range(1, 11):
            self.client.post(
                "/datasets/court_orders/records",
                json={
                    "id": f"r{idx}",
                    "text": f"text-{idx}",
                    "language": "eng_Latn",
                    "provenance": {
                        "source_text": "Manual registry export",
                        "source_type": "manual_entry",
                    },
                },
            )

        response = self.client.post(
            "/datasets/court_orders/splits/sample",
            json={
                "splits": {"dev": 3, "test": 3, "eval": 2},
                "order": "shuffle",
                "seed": 12,
                "disjoint": True,
            },
        )
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        all_ids = []
        for split_name in ["dev", "test", "eval"]:
            ids = [item["id"] for item in body["splits"][split_name]]
            all_ids.extend(ids)
        self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
    unittest.main()
