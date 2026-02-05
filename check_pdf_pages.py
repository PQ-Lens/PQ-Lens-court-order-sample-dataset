#!/usr/bin/env python3
"""Check that PDFs have the expected page count."""

import argparse
import csv
import os
import sys


def iter_pdf_files(root, recursive=True, include_hidden=False):
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            if not include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if not include_hidden and name.startswith("."):
                    continue
                if name.lower().endswith(".pdf"):
                    yield os.path.join(dirpath, name)
    else:
        for name in os.listdir(root):
            if not include_hidden and name.startswith("."):
                continue
            if name.lower().endswith(".pdf"):
                yield os.path.join(root, name)


def load_pypdf():
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        print(
            "Missing dependency 'pypdf'. Install in your venv with:\n"
            "  . .venv/bin/activate && python -m pip install pypdf\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return PdfReader


def main():
    parser = argparse.ArgumentParser(description="Verify PDF page counts")
    parser.add_argument(
        "--dir",
        default=os.getcwd(),
        help="Directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=2,
        help="Expected page count (default: 2)",
    )
    parser.add_argument(
        "--report",
        default="page_check.csv",
        help="CSV report filename (relative to --dir)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subdirectories",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files/directories",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 even if mismatches/errors are found",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    report_path = args.report
    if not os.path.isabs(report_path):
        report_path = os.path.join(root, report_path)

    PdfReader = load_pypdf()

    pdf_files = sorted(iter_pdf_files(root, recursive=not args.no_recursive, include_hidden=args.include_hidden))

    total = 0
    ok = 0
    mismatched = 0
    errors = 0

    rows = []

    for path in pdf_files:
        total += 1
        rel_path = os.path.relpath(path, root)
        try:
            reader = PdfReader(path)
            pages = len(reader.pages)
            status = "ok" if pages == args.expected else "mismatch"
            if status == "ok":
                ok += 1
            else:
                mismatched += 1
            rows.append(
                {
                    "path": rel_path,
                    "pages": pages,
                    "status": status,
                    "error": "",
                }
            )
            if status == "mismatch":
                print(f"Mismatch: {rel_path} -> {pages} pages")
        except Exception as exc:
            errors += 1
            rows.append(
                {
                    "path": rel_path,
                    "pages": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"Error: {rel_path} -> {exc}")

    try:
        with open(report_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "pages", "status", "error"])
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"Failed to write report: {exc}", file=sys.stderr)

    print(
        "Summary: "
        f"total={total} ok={ok} mismatched={mismatched} errors={errors} expected={args.expected}"
    )
    print(f"Report: {report_path}")

    if args.no_fail:
        return 0
    if mismatched or errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
