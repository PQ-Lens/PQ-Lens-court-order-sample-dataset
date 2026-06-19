from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    {
        "id": "legal_translation_mlt_to_eng",
        "messages": [
            {
                "role": "user",
                "content": "Translate this Maltese legal notice to English: Il-Qorti ordnat li l-avviz jigi ppubblikat fir-Registru Pubbliku.",
            }
        ],
    }
]


def _load_prompts(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_PROMPTS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a list of prompt objects")
    return data


def _build_prompt(tokenizer: Any, item: dict[str, Any]) -> str:
    messages = item.get("messages")
    if messages and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if messages:
        return "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)
    return str(item["prompt"])


def run_generation_probe(
    model_id_or_path: str,
    output_path: str | Path,
    *,
    adapter_path: str | None = None,
    prompts_path: str | None = None,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    prompts = _load_prompts(prompts_path)
    results = []
    for item in prompts:
        prompt = _build_prompt(tokenizer, item)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        results.append({"id": item.get("id", "prompt"), "prompt": prompt, "generated": generated.strip()})

    payload = {"model": model_id_or_path, "adapter": adapter_path, "results": results}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic generation probes for a model or adapter.")
    parser.add_argument("--model", required=True, help="Base model id or local merged model path.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--adapter", help="Optional PEFT adapter path.")
    parser.add_argument("--prompts", help="Optional prompts JSON file.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args(argv)

    result = run_generation_probe(
        args.model,
        args.output,
        adapter_path=args.adapter,
        prompts_path=args.prompts,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
