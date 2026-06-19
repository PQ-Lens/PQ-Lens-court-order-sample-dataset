from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .data.schemas import TrainingExample


class TrainingError(RuntimeError):
    """Raised when a training method cannot run."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _summarize_splits(splits: dict[str, list[TrainingExample]]) -> dict[str, Any]:
    summary = {}
    for name, rows in splits.items():
        token_proxy = [max(1, len(row.training_text().split())) for row in rows]
        summary[name] = {
            "examples": len(rows),
            "avg_word_count": round(sum(token_proxy) / len(token_proxy), 2) if token_proxy else 0,
            "max_word_count": max(token_proxy) if token_proxy else 0,
            "groups": len({row.group_id for row in rows}),
        }
    return summary


def run_dry_training(
    config: dict[str, Any],
    splits: dict[str, list[TrainingExample]],
    output_dir: Path,
) -> dict[str, Any]:
    method = config["method"]["name"]
    if method not in {"full", "lora", "qlora"}:
        raise TrainingError(f"Unsupported fine-tuning method: {method}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metrics = {
        "dry_run": True,
        "method": method,
        "model_id": config["model"]["id"],
        "dataset_id": config["dataset"]["id"],
        "split_summary": _summarize_splits(splits),
        "trainable_parameter_policy": "all" if method == "full" else "adapter_only",
        "duration_seconds": round(time.time() - started, 4),
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "resolved_config.json", config)

    train_preview = [row.to_record() for row in splits.get("train", [])[:3]]
    _write_json(output_dir / "train_preview.json", {"examples": train_preview})
    return metrics


def _apply_peft(model: Any, method_config: dict[str, Any]) -> Any:
    method = method_config["name"]
    if method == "full":
        return model

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise TrainingError("LoRA/QLoRA training requires peft.") from exc

    if method == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora = method_config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get("target_modules", "all-linear"),
        bias=lora.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def run_real_training(
    config: dict[str, Any],
    splits: dict[str, list[TrainingExample]],
    model: Any,
    tokenizer: Any,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise TrainingError("Real training requires datasets, transformers, and trl.") from exc

    model = _apply_peft(model, config["method"])
    train_dataset = Dataset.from_list([{"text": row.training_text()} for row in splits["train"]])
    eval_rows = splits.get("validation") or splits.get("test") or []
    eval_dataset = Dataset.from_list([{"text": row.training_text()} for row in eval_rows]) if eval_rows else None

    training = config.get("training", {})
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        logging_steps=int(training.get("logging_steps", 10)),
        save_strategy=training.get("save_strategy", "epoch"),
        evaluation_strategy="epoch" if eval_dataset else "no",
        bf16=bool(training.get("bf16", False)),
        fp16=bool(training.get("fp16", False)),
        report_to=training.get("report_to", "none"),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=int(config["model"].get("max_seq_length", 2048)),
        args=args,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    metrics = {"dry_run": False, "train_result": trainer.state.log_history, "split_summary": _summarize_splits(splits)}
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "resolved_config.json", config)
    return metrics
