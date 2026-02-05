#!/usr/bin/env python3
"""Scrape court-order PDFs into a local folder (standard library only)."""

import argparse
import csv
import os
import re
import sys
import tempfile
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

USER_AGENT = "Mozilla/5.0 (compatible; court-orders-scraper/1.0)"
TIMEOUT_SECS = 20
CHUNK_SIZE = 64 * 1024
PAGINATION_RE = re.compile(r"/court-orders/page/\d+/?$", re.IGNORECASE)


class HrefParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def fetch_text(url):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=TIMEOUT_SECS) as resp:
        content_type = resp.headers.get("Content-Type", "")
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip() or "utf-8"
        data = resp.read()
    try:
        return data.decode(charset, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def extract_hrefs(html):
    parser = HrefParser()
    parser.feed(html)
    return parser.hrefs


def is_pdf_url(url):
    try:
        path = urlparse(url).path
    except ValueError:
        return False
    return path.lower().endswith(".pdf")


def is_pagination_url(url, base_netloc):
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.netloc != base_netloc:
        return False
    return bool(PAGINATION_RE.search(parsed.path))


def unique_filename(base_name, name_to_url, out_dir):
    candidate = base_name
    if candidate not in name_to_url and not os.path.exists(os.path.join(out_dir, candidate)):
        return candidate

    stem, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".pdf"
    idx = 1
    while True:
        candidate = f"{stem}__dup{idx}{ext}"
        if candidate not in name_to_url and not os.path.exists(os.path.join(out_dir, candidate)):
            return candidate
        idx += 1


def load_manifest(manifest_path):
    url_to_name = {}
    name_to_url = {}
    if not os.path.exists(manifest_path):
        return url_to_name, name_to_url
    try:
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                url = (row.get("url") or "").strip()
                filename = (row.get("filename") or "").strip()
                if url and filename:
                    url_to_name[url] = filename
                    name_to_url[filename] = url
    except OSError:
        pass
    return url_to_name, name_to_url


def download_file(url, dest_path):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    temp_handle = None
    temp_path = None
    total = 0
    try:
        with urlopen(req, timeout=TIMEOUT_SECS) as resp:
            temp_handle = tempfile.NamedTemporaryFile(
                delete=False,
                dir=os.path.dirname(dest_path),
                prefix=os.path.basename(dest_path) + ".",
                suffix=".part",
            )
            temp_path = temp_handle.name
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                temp_handle.write(chunk)
                total += len(chunk)
    finally:
        if temp_handle is not None:
            temp_handle.close()
    os.replace(temp_path, dest_path)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Download PDFs linked from the court-orders page"
    )
    parser.add_argument(
        "--out-dir",
        default=os.getcwd(),
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--base-url",
        default="https://assetrecovery.mt/court-orders/",
        help="Base court-orders URL",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between downloads in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.csv",
        help="Manifest CSV filename (relative to out-dir)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = args.manifest
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(out_dir, manifest_path)

    print(f"Output directory: {out_dir}")
    print(f"Base URL: {args.base_url}")

    try:
        base_html = fetch_text(args.base_url)
    except (HTTPError, URLError, ValueError) as exc:
        print(f"Failed to fetch base URL: {exc}", file=sys.stderr)
        return 1

    base_hrefs = extract_hrefs(base_html)
    base_netloc = urlparse(args.base_url).netloc

    page_urls = []
    seen_pages = set()

    def add_page(url):
        if url in seen_pages:
            return
        seen_pages.add(url)
        page_urls.append(url)

    add_page(args.base_url)

    for href in base_hrefs:
        abs_url = urljoin(args.base_url, href)
        if is_pagination_url(abs_url, base_netloc):
            add_page(abs_url)

    print(f"Pages to scan: {len(page_urls)}")

    pdf_urls = []
    seen_pdfs = set()

    def add_pdf(url):
        if url in seen_pdfs:
            return
        seen_pdfs.add(url)
        pdf_urls.append(url)

    for page_url in page_urls:
        try:
            html = fetch_text(page_url)
        except (HTTPError, URLError, ValueError) as exc:
            print(f"Failed to fetch page {page_url}: {exc}", file=sys.stderr)
            continue
        for href in extract_hrefs(html):
            abs_url = urljoin(page_url, href)
            if is_pdf_url(abs_url):
                add_pdf(abs_url)

    print(f"Unique PDF links found: {len(pdf_urls)}")

    url_to_name, name_to_url = load_manifest(manifest_path)

    results = []
    downloaded = 0
    skipped = 0
    failed = 0

    for idx, url in enumerate(pdf_urls, start=1):
        if url in url_to_name:
            filename = url_to_name[url]
        else:
            base_name = os.path.basename(urlparse(url).path)
            if not base_name:
                base_name = f"download_{idx}.pdf"
            filename = unique_filename(base_name, name_to_url, out_dir)
            url_to_name[url] = filename
            name_to_url[filename] = url

        dest_path = os.path.join(out_dir, filename)

        if os.path.exists(dest_path):
            size = os.path.getsize(dest_path)
            results.append(
                {"url": url, "filename": filename, "status": "skipped", "bytes": size, "error": ""}
            )
            skipped += 1
            print(f"[{idx}/{len(pdf_urls)}] skip {filename}")
            continue

        try:
            size = download_file(url, dest_path)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            failed += 1
            results.append(
                {"url": url, "filename": filename, "status": "failed", "bytes": 0, "error": str(exc)}
            )
            print(f"[{idx}/{len(pdf_urls)}] fail {filename}: {exc}")
            continue

        downloaded += 1
        results.append(
            {"url": url, "filename": filename, "status": "downloaded", "bytes": size, "error": ""}
        )
        print(f"[{idx}/{len(pdf_urls)}] ok   {filename} ({size} bytes)")
        time.sleep(max(args.delay, 0))

    try:
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["url", "filename", "status", "bytes", "error"])
            writer.writeheader()
            for row in results:
                writer.writerow(row)
    except OSError as exc:
        print(f"Failed to write manifest: {exc}", file=sys.stderr)

    print(
        "Summary: "
        f"total={len(pdf_urls)} downloaded={downloaded} skipped={skipped} failed={failed}"
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
