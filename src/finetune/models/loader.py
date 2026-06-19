from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DryRunModel:
    model_id: str
    method: str


@dataclass
class DryRunTokenizer:
    model_id: str
    eos_token: str = "</s>"
    pad_token: str = "</s>"


def load_model_and_tokenizer(model_config: dict[str, Any], method_config: dict[str, Any], *, dry_run: bool = False):
    model_id = model_config["id"]
    method = method_config.get("name", "full")
    if dry_run:
        return DryRunModel(model_id=model_id, method=method), DryRunTokenizer(model_id=model_id)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Real training requires torch and transformers. Use --dry-run for validation.") from exc

    dtype_name = model_config.get("dtype", "bf16")
    dtype = torch.bfloat16 if dtype_name == "bf16" and torch.cuda.is_available() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=bool(model_config.get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=method_config.get("quantization", {}).get("type", "nf4"),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=bool(method_config.get("quantization", {}).get("double_quant", True)),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype if quantization_config is None else None,
        device_map=model_config.get("device_map", "auto"),
        quantization_config=quantization_config,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer
