# PQ-Lens Fine-Tuning Pipeline

This repository now includes a config-driven fine-tuning pipeline for supervised model adaptation. It is designed so the model, dataset, and fine-tuning method can be swapped independently.

## Architecture

```text
[Dataset Sources]
    |
    |-- local JSONL / CSV / Parquet / XLSX
    |-- Hugging Face datasets
    |-- PQ-Lens Flask API
    |
    v
[Dataset Loader]
    |
    v
[Canonical TrainingExample Schema]
    |
    |-- id
    |-- prompt
    |-- completion
    |-- text
    |-- messages
    |-- source_lang / target_lang
    |-- group_id
    |-- metadata / provenance
    |
    v
[Preprocessing]
    |
    |-- deduplication
    |-- grouped train / validation / test split
    |-- prompt templating
    |-- max length filtering
    |
    v
[Model Loader]
    |
    |-- AutoTokenizer
    |-- AutoModelForCausalLM
    |-- optional 4-bit BitsAndBytesConfig
    |
    v
[Method Selector]
    |
    |-- full: train all model weights
    |-- lora: freeze base model and train PEFT adapters
    |-- qlora: 4-bit base model plus PEFT adapters
    |
    v
[Trainer]
    |
    |-- dry-run validator
    |-- TRL SFTTrainer for real training
    |
    v
[Artifacts]
    |
    |-- metrics.json
    |-- resolved_config.json
    |-- train_preview.json
    |-- model or adapter weights for real runs
```

## Dry Run

The dry run validates the pipeline without downloading a model or requiring a GPU:

```bash
python3 scripts/train_finetune.py \
  --config configs/experiments/dry_run_qlora_court_orders.json \
  --dry-run \
  --output-dir artifacts/finetune/dry_run_local
```

Expected artifacts:

- `metrics.json`
- `resolved_config.json`
- `train_preview.json`

## Gemma 3 1B Smoke Test

Before running the real Gemma smoke workflow, verify that the Hugging Face account behind the cached token can download the gated checkpoint:

```bash
python3 scripts/check_hf_model_access.py \
  --model google/gemma-3-1b-it \
  --output artifacts/finetune/gemma3_1b_qlora_smoke/hf_access.json
```

Only continue if `"ok": true`. A gated-repo `403` means the account still needs access approval on the model page.

Then run the baseline prompt probe, QLoRA smoke fine-tune, adapter probe, merge/export, and Ollama import:

```bash
python3 scripts/run_generation_probe.py \
  --model google/gemma-3-1b-it \
  --prompts data/training_samples/gemma_probe_prompts.json \
  --output artifacts/finetune/gemma3_1b_qlora_smoke/baseline_generations.json

python3 scripts/train_finetune.py \
  --config configs/experiments/gemma3_1b_qlora_smoke.json \
  --output-dir artifacts/finetune/gemma3_1b_qlora_smoke/adapter

python3 scripts/run_generation_probe.py \
  --model google/gemma-3-1b-it \
  --adapter artifacts/finetune/gemma3_1b_qlora_smoke/adapter \
  --prompts data/training_samples/gemma_probe_prompts.json \
  --output artifacts/finetune/gemma3_1b_qlora_smoke/adapter_generations.json

python3 scripts/merge_adapter_for_ollama.py \
  --base-model google/gemma-3-1b-it \
  --adapter artifacts/finetune/gemma3_1b_qlora_smoke/adapter \
  --output-dir artifacts/finetune/gemma3_1b_qlora_smoke/merged \
  --modelfile artifacts/finetune/gemma3_1b_qlora_smoke/Modelfile

ollama create pqlens-gemma3-1b-legal-smoke:latest \
  -f artifacts/finetune/gemma3_1b_qlora_smoke/Modelfile
```

## Real Training

Install the heavier ML stack only on machines that will run training:

```bash
pip install -r requirements-ml.txt
```

Then run without `--dry-run`:

```bash
python3 scripts/train_finetune.py \
  --config configs/experiments/dry_run_qlora_court_orders.json \
  --output-dir artifacts/finetune/qlora_run
```

For production experiments, copy the dry-run model config and replace `model.id` with a real Hugging Face model such as a Llama, Mistral, Qwen, or Gemma checkpoint. Keep model-specific LoRA target modules in the method config.

## Method Parameters

Full fine-tuning:

```json
{
  "learning_rate": 0.00001,
  "num_train_epochs": 1,
  "gradient_checkpointing": true,
  "weight_decay": 0.01
}
```

LoRA:

```json
{
  "r": 16,
  "alpha": 32,
  "dropout": 0.05,
  "target_modules": "all-linear"
}
```

QLoRA:

```json
{
  "quantization": {
    "bits": 4,
    "type": "nf4",
    "double_quant": true
  },
  "lora": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": "all-linear"
  }
}
```

These defaults follow the LoRA and QLoRA papers' practical setup: adapter-only training for LoRA, and NF4 4-bit quantization plus LoRA adapters for QLoRA.

## Edge Cases Covered

- Empty datasets are rejected.
- Invalid JSONL rows include file and line number in the error.
- Split ratios must sum to 1.0.
- Training split cannot be empty.
- Deduplication runs before splitting to avoid repeated examples leaking into evaluation.
- Splits are made by `group_id`, so rows from the same source document stay in the same split.
- Dry-run mode does not import `torch`, `transformers`, `peft`, `trl`, or `bitsandbytes`.
- API loading paginates through `/datasets/{id}/records`.

## Relevant Research and Tooling

- LoRA: <https://arxiv.org/abs/2106.09685>
- QLoRA: <https://arxiv.org/abs/2305.14314>
- Hugging Face PEFT LoRA docs: <https://huggingface.co/docs/peft/package_reference/lora>
- Hugging Face PEFT quantization docs: <https://huggingface.co/docs/peft/en/developer_guides/quantization>
- TRL SFTTrainer: <https://huggingface.co/docs/trl/en/sft_trainer>
- Hugging Face Datasets: <https://huggingface.co/docs/datasets/en/index>
