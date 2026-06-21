from __future__ import annotations

import json
import inspect
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


def _sft_record(example: TrainingExample) -> dict[str, Any]:
    if example.messages:
        return {"messages": example.messages}
    return {"text": example.training_text()}


def _trainable_parameter_summary(model: Any) -> dict[str, Any]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    percent = round((trainable / total) * 100, 4) if total else 0.0
    return {"trainable": trainable, "total": total, "percent": percent}


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
    lora_kwargs = {
        "r": int(lora.get("r", 16)),
        "lora_alpha": int(lora.get("alpha", 32)),
        "lora_dropout": float(lora.get("dropout", 0.05)),
        "bias": lora.get("bias", "none"),
        "task_type": "CAUSAL_LM",
    }
    if lora.get("target_modules"):
        lora_kwargs["target_modules"] = lora["target_modules"]
    if lora.get("modules_to_save"):
        lora_kwargs["modules_to_save"] = lora["modules_to_save"]
    if "ensure_weight_tying" in inspect.signature(LoraConfig).parameters:
        lora_kwargs["ensure_weight_tying"] = bool(lora.get("ensure_weight_tying", False))
    peft_config = LoraConfig(**lora_kwargs)
    return get_peft_model(model, peft_config)


def _training_arguments_kwargs(training: dict[str, Any], output_dir: Path, has_eval: bool) -> dict[str, Any]:
    try:
        from transformers import TrainingArguments
    except ImportError as exc:  # pragma: no cover - checked by caller
        raise TrainingError("Real training requires transformers.") from exc

    params = inspect.signature(TrainingArguments).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training.get("learning_rate", 2e-4)),
        "num_train_epochs": float(training.get("num_train_epochs", 1)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "logging_steps": int(training.get("logging_steps", 10)),
        "save_strategy": training.get("save_strategy", "epoch"),
        "bf16": bool(training.get("bf16", False)),
        "fp16": bool(training.get("fp16", False)),
        "report_to": training.get("report_to", "none"),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
    }
    if training.get("max_steps") is not None:
        kwargs["max_steps"] = int(training["max_steps"])
    eval_value = "steps" if has_eval and training.get("eval_steps") else ("epoch" if has_eval else "no")
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    else:
        kwargs["evaluation_strategy"] = eval_value
    if training.get("eval_steps") is not None:
        kwargs["eval_steps"] = int(training["eval_steps"])
    return kwargs


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
    train_uses_messages = any(row.messages for row in splits["train"])
    train_dataset = Dataset.from_list([_sft_record(row) for row in splits["train"]])
    eval_rows = splits.get("validation") or splits.get("test") or []
    eval_dataset = Dataset.from_list([_sft_record(row) for row in eval_rows]) if eval_rows else None

    training = config.get("training", {})
    args = TrainingArguments(**_training_arguments_kwargs(training, output_dir, eval_dataset is not None))

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": args,
    }
    trainer_params = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params and not train_uses_messages:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = int(config["model"].get("max_seq_length", 2048))

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    metrics = {
        "dry_run": False,
        "train_result": trainer.state.log_history,
        "split_summary": _summarize_splits(splits),
        "parameters": _trainable_parameter_summary(model),
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "resolved_config.json", config)
    return metrics
