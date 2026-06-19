from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_experiment_config, require_section
from .data import load_dataset_examples, prepare_splits
from .models import load_model_and_tokenizer
from .training import run_dry_training, run_real_training


def resolve_output_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    output_root = Path(config.get("output_root", "artifacts/finetune")).expanduser()
    experiment_id = config.get("experiment_id", "experiment")
    return (Path.cwd() / output_root / experiment_id).resolve()


def run_pipeline(config_path: str | Path, *, dry_run: bool = False, output_dir: str | None = None) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    require_section(config, "model")
    require_section(config, "dataset")
    require_section(config, "method")
    config.setdefault("training", {})
    config["runtime"] = {"dry_run": dry_run}

    examples = load_dataset_examples(config["dataset"])
    splits = prepare_splits(examples, config["dataset"])
    output_path = resolve_output_dir(config, output_dir)

    model, tokenizer = load_model_and_tokenizer(config["model"], config["method"], dry_run=dry_run)
    if dry_run:
        metrics = run_dry_training(config, splits, output_path)
    else:
        metrics = run_real_training(config, splits, model, tokenizer, output_path)
    return {"output_dir": str(output_path), "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a PQ-Lens fine-tuning experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment config JSON/YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the pipeline without loading ML dependencies.")
    parser.add_argument("--output-dir", help="Override artifact output directory.")
    args = parser.parse_args(argv)

    result = run_pipeline(args.config, dry_run=args.dry_run, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
