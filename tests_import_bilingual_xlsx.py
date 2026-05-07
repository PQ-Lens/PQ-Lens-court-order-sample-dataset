import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from scripts.import_bilingual_xlsx import (
    DATASET_DESCRIPTION,
    DATASET_ID,
    DATASET_SUMMARY,
    LANGUAGE_PAIR,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    WORKSHEET_NAME,
    build_record_payload,
    dataset_payload,
    load_rows,
    verify_inserted,
)


class BilingualImporterTestCase(unittest.TestCase):
    def _write_workbook(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "bilingual.xlsx"

        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = WORKSHEET_NAME
        worksheet.append(["English", "Maltese"])
        worksheet.append(["English notice", "Avviż bil-Malti"])
        worksheet.append(["English only", ""])
        worksheet.append(["Another English notice", "Avviż ieħor bil-Malti"])
        workbook.save(path)
        return path

    def test_load_rows_returns_complete_bilingual_pairs_only(self) -> None:
        rows = load_rows(self._write_workbook())

        self.assertEqual([row.row_number for row in rows], [2, 4])
        self.assertEqual([row.record_id for row in rows], ["bilingual_row_0002", "bilingual_row_0004"])

    def test_dataset_payload_uses_tidy_metadata_and_readme_conventions(self) -> None:
        payload = dataset_payload()

        self.assertEqual(payload["id"], DATASET_ID)
        self.assertEqual(payload["summary"], DATASET_SUMMARY)
        self.assertEqual(payload["description"], DATASET_DESCRIPTION)
        self.assertEqual(payload["language_mode"], "bilingual_aligned")
        self.assertEqual(payload["synthetic_status"], "non_synthetic")
        self.assertEqual(payload["provenance"]["source_type"], "web")

    def test_build_record_payload_sets_translation_metadata(self) -> None:
        row = load_rows(self._write_workbook())[0]
        payload = build_record_payload(row, Path("bilingual.xlsx"))

        self.assertEqual(payload["id"], "bilingual_row_0002")
        self.assertEqual(payload["language"], SOURCE_LANGUAGE)
        self.assertEqual(payload["language_pair"], LANGUAGE_PAIR)
        self.assertEqual(payload["translation_metadata"]["source_language"], SOURCE_LANGUAGE)
        self.assertEqual(payload["translation_metadata"]["target_language"], TARGET_LANGUAGE)
        self.assertEqual(payload["translation_metadata"]["target_text"], "Avviż bil-Malti")
        self.assertEqual(payload["attributes"]["row_number"], 2)

    def test_verify_inserted_accepts_matching_api_records(self) -> None:
        rows = load_rows(self._write_workbook())
        actual_records = [build_record_payload(row, Path("bilingual.xlsx")) for row in rows]

        verify_inserted(rows, actual_records)


if __name__ == "__main__":
    unittest.main()
