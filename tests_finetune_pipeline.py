import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.config import load_experiment_config
from finetune.data.loaders import normalize_rows
from finetune.data.preprocess import PreprocessError, prepare_splits
from finetune.train import run_pipeline


class FineTunePipelineTestCase(unittest.TestCase):
    def test_experiment_config_composes_model_dataset_and_method(self) -> None:
        config = load_experiment_config("configs/experiments/dry_run_qlora_court_orders.json")

        self.assertEqual(config["model"]["id"], "dry-run/tiny-causal-lm")
        self.assertEqual(config["dataset"]["id"], "court_orders_tiny")
        self.assertEqual(config["method"]["name"], "qlora")
        self.assertEqual(config["method"]["quantization"]["type"], "nf4")

    def test_prepare_splits_deduplicates_and_keeps_groups_disjoint(self) -> None:
        rows = [
            {
                "id": "a1",
                "text": "same legal text",
                "translation_metadata": {"target_text": "same translation"},
                "metadata": {"document_id": "doc-a"},
            },
            {
                "id": "a2",
                "text": "same legal text",
                "translation_metadata": {"target_text": "same translation"},
                "metadata": {"document_id": "doc-a"},
            },
            {
                "id": "b1",
                "text": "different text",
                "translation_metadata": {"target_text": "different translation"},
                "metadata": {"document_id": "doc-b"},
            },
            {
                "id": "c1",
                "text": "third text",
                "translation_metadata": {"target_text": "third translation"},
                "metadata": {"document_id": "doc-c"},
            },
        ]
        examples = normalize_rows(
            rows,
            {
                "columns": {
                    "id": "id",
                    "text": "text",
                    "completion": "translation_metadata.target_text",
                    "group_id": "metadata.document_id",
                }
            },
        )
        splits = prepare_splits(
            examples,
            {
                "deduplicate": True,
                "template": {
                    "force_prompt_completion": True,
                    "prompt": "Translate:\\n{text}",
                    "completion": "{completion}",
                },
                "split": {"seed": 7, "ratios": {"train": 0.34, "validation": 0.33, "test": 0.33}},
            },
        )

        all_ids = [row.id for split in splits.values() for row in split]
        self.assertEqual(len(all_ids), 3)

        group_to_split = {}
        for split_name, split_rows in splits.items():
            for row in split_rows:
                existing = group_to_split.setdefault(row.group_id, split_name)
                self.assertEqual(existing, split_name)

    def test_invalid_split_ratios_are_rejected(self) -> None:
        examples = normalize_rows([{"id": "r1", "text": "hello"}], {"columns": {"id": "id", "text": "text"}})

        with self.assertRaises(PreprocessError):
            prepare_splits(
                examples,
                {
                    "split": {"ratios": {"train": 0.9, "test": 0.9}},
                },
            )

    def test_dry_run_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_pipeline(
                "configs/experiments/dry_run_qlora_court_orders.json",
                dry_run=True,
                output_dir=temp_dir,
            )
            output_dir = Path(result["output_dir"])

            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "resolved_config.json").exists())
            self.assertTrue((output_dir / "train_preview.json").exists())

            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["dry_run"])
            self.assertEqual(metrics["method"], "qlora")
            self.assertEqual(metrics["trainable_parameter_policy"], "adapter_only")
            self.assertGreater(metrics["split_summary"]["train"]["examples"], 0)


if __name__ == "__main__":
    unittest.main()
