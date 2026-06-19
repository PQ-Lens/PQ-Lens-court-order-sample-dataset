from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any

from .schemas import TrainingExample


class PreprocessError(ValueError):
    """Raised when examples cannot be prepared for training."""


def _fingerprint(example: TrainingExample) -> str:
    text = " ".join(example.training_text().lower().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate(examples: list[TrainingExample]) -> list[TrainingExample]:
    seen: set[str] = set()
    unique = []
    for example in examples:
        key = _fingerprint(example)
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def _format_example(example: TrainingExample, template: dict[str, Any]) -> TrainingExample:
    task = template.get("task", "instruction")
    if example.text and not template.get("force_prompt_completion", False):
        return example
    if example.messages and task == "chat":
        return example

    prompt_template = template.get("prompt")
    completion_template = template.get("completion", "{completion}")

    if prompt_template:
        prompt = prompt_template.format(
            prompt=example.prompt,
            completion=example.completion,
            text=example.text,
            source_lang=example.source_lang or "",
            target_lang=example.target_lang or "",
        )
    elif example.prompt:
        prompt = example.prompt
    elif task == "translation":
        prompt = f"Translate from {example.source_lang or 'source'} to {example.target_lang or 'target'}:\n{example.text}"
    else:
        prompt = example.text

    completion = completion_template.format(
        prompt=example.prompt,
        completion=example.completion,
        text=example.text,
        source_lang=example.source_lang or "",
        target_lang=example.target_lang or "",
    )

    formatted_text = template.get("text")
    if formatted_text:
        text = formatted_text.format(prompt=prompt, completion=completion)
    else:
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{completion}".strip()

    return TrainingExample(
        id=example.id,
        prompt=prompt,
        completion=completion,
        text=text,
        messages=example.messages,
        source_lang=example.source_lang,
        target_lang=example.target_lang,
        group_id=example.group_id,
        metadata=example.metadata,
    )


def _filter_length(examples: list[TrainingExample], max_chars: int | None) -> list[TrainingExample]:
    if not max_chars:
        return examples
    return [example for example in examples if len(example.training_text()) <= max_chars]


def split_by_group(
    examples: list[TrainingExample],
    split_config: dict[str, Any],
) -> dict[str, list[TrainingExample]]:
    ratios = split_config.get("ratios", {"train": 0.8, "validation": 0.1, "test": 0.1})
    if not ratios or abs(sum(float(v) for v in ratios.values()) - 1.0) > 0.001:
        raise PreprocessError("Split ratios must sum to 1.0")

    grouped: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        grouped[example.group_id or example.id].append(example)

    groups = list(grouped.items())
    random.Random(int(split_config.get("seed", 42))).shuffle(groups)

    total = len(groups)
    names = list(ratios.keys())
    cutoffs: dict[str, int] = {}
    consumed = 0
    for name in names[:-1]:
        count = int(round(total * float(ratios[name])))
        cutoffs[name] = count
        consumed += count
    cutoffs[names[-1]] = max(total - consumed, 0)

    splits: dict[str, list[TrainingExample]] = {name: [] for name in names}
    cursor = 0
    for name in names:
        count = cutoffs[name]
        for _, rows in groups[cursor : cursor + count]:
            splits[name].extend(rows)
        cursor += count

    if not splits.get("train"):
        raise PreprocessError("Training split is empty after preprocessing")
    return splits


def prepare_splits(
    examples: list[TrainingExample],
    dataset_config: dict[str, Any],
) -> dict[str, list[TrainingExample]]:
    if dataset_config.get("deduplicate", True):
        examples = deduplicate(examples)

    examples = [_format_example(example, dataset_config.get("template", {})) for example in examples]
    examples = _filter_length(examples, dataset_config.get("max_chars"))

    if not examples:
        raise PreprocessError("No examples remain after filtering")

    return split_by_group(examples, dataset_config.get("split", {}))
