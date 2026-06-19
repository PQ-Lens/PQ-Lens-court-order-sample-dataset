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
from finetune.export import write_ollama_modelfile
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

    def test_gemma_smoke_config_uses_bounded_qlora_and_messages(self) -> None:
        config = load_experiment_config("configs/experiments/gemma3_1b_qlora_smoke.json")

        self.assertEqual(config["model"]["id"], "google/gemma-3-1b-it")
        self.assertEqual(config["method"]["name"], "qlora")
        self.assertEqual(config["method"]["lora"]["r"], 8)
        self.assertEqual(config["training"]["max_steps"], 10)
        self.assertEqual(config["dataset"]["columns"]["messages"], "messages")

    def test_message_rows_survive_preprocessing_for_chat_template_training(self) -> None:
        rows = [
            {
                "id": "chat-1",
                "messages": [
                    {"role": "user", "content": "Translate: Avviz"},
                    {"role": "assistant", "content": "Notice"},
                ],
                "metadata": {"document_id": "doc-chat-1"},
            },
            {
                "id": "chat-2",
                "messages": [
                    {"role": "user", "content": "Translate: Qorti"},
                    {"role": "assistant", "content": "Court"},
                ],
                "metadata": {"document_id": "doc-chat-2"},
            },
        ]
        examples = normalize_rows(
            rows,
            {"columns": {"id": "id", "messages": "messages", "group_id": "metadata.document_id"}},
        )
        splits = prepare_splits(examples, {"split": {"ratios": {"train": 0.5, "test": 0.5}}})

        all_examples = [row for split_rows in splits.values() for row in split_rows]
        self.assertTrue(all_examples)
        self.assertTrue(all_examples[0].messages)
        self.assertIn("user", {message["role"] for message in all_examples[0].messages})

    def test_ollama_modelfile_points_to_merged_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "merged"
            model_dir.mkdir()
            modelfile = Path(temp_dir) / "Modelfile"

            write_ollama_modelfile(model_dir, modelfile, temperature=0.1, num_ctx=1024)

            text = modelfile.read_text(encoding="utf-8")
            self.assertIn(f"FROM {model_dir.resolve()}", text)
            self.assertIn("PARAMETER temperature 0.1", text)
            self.assertIn("PARAMETER num_ctx 1024", text)


if __name__ == "__main__":
    unittest.main()
