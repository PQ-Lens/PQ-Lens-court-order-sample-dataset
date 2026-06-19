from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_model_access(model_id: str, *, filename: str = "config.json") -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    result: dict[str, Any] = {"model_id": model_id, "filename": filename}
    try:
        who = api.whoami()
        result["authenticated"] = True
        result["user"] = who.get("name") or who.get("email") or "unknown"
    except Exception as exc:
        result.update({"authenticated": False, "ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        return result

    try:
        info = api.model_info(model_id)
        result["gated"] = getattr(info, "gated", None)
    except Exception as exc:
        result.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        return result

    try:
        path = hf_hub_download(model_id, filename)
        result.update({"ok": True, "resolved_path": path})
    except Exception as exc:
        result.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc).splitlines()[0]})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Hugging Face model access before training.")
    parser.add_argument("--model", required=True, help="Hugging Face model id to check.")
    parser.add_argument("--filename", default="config.json", help="Small file to test-download.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    result = check_model_access(args.model, filename=args.filename)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
