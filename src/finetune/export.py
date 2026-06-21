from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_adapter(base_model: str, adapter_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    result = {"base_model": base_model, "adapter_path": str(adapter_path), "output_dir": str(output_dir)}
    (output_dir / "merge_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def write_ollama_modelfile(
    model_path: str | Path,
    output_path: str | Path,
    *,
    temperature: float = 0.2,
    num_ctx: int = 2048,
) -> Path:
    model_path = Path(model_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"FROM {model_path}",
                'TEMPLATE """<bos>{{ if .System }}<start_of_turn>user',
                "{{ .System }}",
                "",
                "{{ .Prompt }}<end_of_turn>",
                "{{ else }}<start_of_turn>user",
                "{{ .Prompt }}<end_of_turn>",
                "{{ end }}<start_of_turn>model",
                '"""',
                f"PARAMETER temperature {temperature}",
                f"PARAMETER num_ctx {num_ctx}",
                'PARAMETER stop "<end_of_turn>"',
                'SYSTEM """You are a legal translation assistant for Maltese and English court-order text."""',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter and prepare Ollama import assets.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modelfile", help="Optional Modelfile path to create for Ollama.")
    args = parser.parse_args(argv)

    result = merge_adapter(args.base_model, args.adapter, args.output_dir)
    if args.modelfile:
        result["modelfile"] = str(write_ollama_modelfile(args.output_dir, args.modelfile))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
