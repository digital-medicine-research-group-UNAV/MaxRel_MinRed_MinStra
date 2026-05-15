#!/usr/bin/env python3
"""Download Bayesian networks from the bnlearn repository.

By default, this script crawls https://www.bnlearn.com/bnrepository/
and downloads all available .bif.gz network files into ./networks,
decompressing each file to .bif.
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

BASE_REPOSITORY_URL = "https://www.bnlearn.com/bnrepository/"
DEFAULT_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (compatible; bnlearn-network-downloader/1.0)"

SUPPORTED_FORMATS = {"bif", "dsc", "net", "rda", "rds"}
FILE_LINK_RE = re.compile(r"\.(bif|dsc|net|rda|rds)\.gz$", re.IGNORECASE)


class AnchorLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)
                break


def fetch_bytes(url: str, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_text(url: str, timeout: int) -> str:
    data = fetch_bytes(url, timeout=timeout)
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def normalize_url(base_url: str, href: str) -> str:
    absolute = urljoin(base_url, href)
    clean, _fragment = urldefrag(absolute)
    return clean


def is_repo_html_page(url: str) -> bool:
    if not url.startswith(BASE_REPOSITORY_URL):
        return False
    parsed = urlparse(url)
    path = parsed.path
    if path.endswith("/"):
        return True
    return path.endswith(".html")


def parse_links(page_url: str, html_text: str) -> list[str]:
    parser = AnchorLinkParser()
    parser.feed(html_text)
    return [normalize_url(page_url, href) for href in parser.hrefs]


def discover_repository_pages(start_url: str, timeout: int) -> list[str]:
    queue = [start_url]
    visited: set[str] = set()
    pages: set[str] = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        try:
            html_text = fetch_text(current, timeout=timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[WARN] Could not crawl page: {current} ({exc})", file=sys.stderr)
            continue

        pages.add(current)

        for link in parse_links(current, html_text):
            if not is_repo_html_page(link):
                continue
            if link not in visited:
                queue.append(link)

    return sorted(pages)


def extract_format_from_url(url: str) -> str | None:
    match = FILE_LINK_RE.search(url)
    if not match:
        return None
    return match.group(1).lower()


def discover_network_files(pages: Iterable[str], selected_formats: set[str], timeout: int) -> list[str]:
    found: set[str] = set()

    for page in pages:
        try:
            html_text = fetch_text(page, timeout=timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[WARN] Could not read page for links: {page} ({exc})", file=sys.stderr)
            continue

        for link in parse_links(page, html_text):
            if not link.startswith(BASE_REPOSITORY_URL):
                continue
            link_format = extract_format_from_url(link)
            if link_format is None:
                continue
            if link_format in selected_formats:
                found.add(link)

    return sorted(found)


def write_file(path: Path, data: bytes, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def save_download(
    file_url: str,
    output_dir: Path,
    keep_gz: bool,
    overwrite: bool,
    timeout: int,
) -> tuple[Path, bool]:
    filename = Path(urlparse(file_url).path).name
    gz_target = output_dir / filename

    raw_data = fetch_bytes(file_url, timeout=timeout)

    if keep_gz:
        changed = write_file(gz_target, raw_data, overwrite=overwrite)
        return gz_target, changed

    if not filename.endswith(".gz"):
        changed = write_file(gz_target, raw_data, overwrite=overwrite)
        return gz_target, changed

    uncompressed_name = filename[:-3]
    uncompressed_target = output_dir / uncompressed_name
    uncompressed_data = gzip.GzipFile(fileobj=io.BytesIO(raw_data)).read()
    changed = write_file(uncompressed_target, uncompressed_data, overwrite=overwrite)
    return uncompressed_target, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Bayesian networks from bnlearn.com/bnrepository."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "networks",
        help="Directory where downloaded files are written.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="bif",
        help="Comma-separated formats to download (bif,dsc,net,rda,rds). Default: bif",
    )
    parser.add_argument(
        "--keep-gz",
        action="store_true",
        help="Keep downloaded .gz files instead of decompressing them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--index-url",
        type=str,
        default=BASE_REPOSITORY_URL,
        help="Repository index URL to crawl.",
    )
    return parser.parse_args()


def parse_formats(formats_arg: str) -> set[str]:
    parsed = {fmt.strip().lower() for fmt in formats_arg.split(",") if fmt.strip()}
    invalid = sorted(parsed - SUPPORTED_FORMATS)
    if invalid:
        raise ValueError(
            f"Unsupported format(s): {', '.join(invalid)}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    if not parsed:
        raise ValueError("At least one format must be specified in --formats.")
    return parsed


def main() -> int:
    args = parse_args()

    try:
        selected_formats = parse_formats(args.formats)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    start_url = normalize_url(BASE_REPOSITORY_URL, args.index_url)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Crawling repository pages from: {start_url}")
    pages = discover_repository_pages(start_url, timeout=args.timeout)
    print(f"[INFO] Found {len(pages)} repository page(s)")

    print(f"[INFO] Discovering network files for formats: {', '.join(sorted(selected_formats))}")
    file_urls = discover_network_files(pages, selected_formats, timeout=args.timeout)
    print(f"[INFO] Found {len(file_urls)} downloadable network file(s)")

    if not file_urls:
        print("[WARN] No network files found. Nothing to download.")
        return 1

    downloaded = 0
    skipped = 0
    failed = 0

    for file_url in file_urls:
        try:
            target, changed = save_download(
                file_url=file_url,
                output_dir=output_dir,
                keep_gz=args.keep_gz,
                overwrite=args.overwrite,
                timeout=args.timeout,
            )
            if changed:
                downloaded += 1
                print(f"[OK] Downloaded: {target.name}")
            else:
                skipped += 1
                print(f"[SKIP] Exists: {target.name}")
        except (HTTPError, URLError, TimeoutError, OSError, gzip.BadGzipFile) as exc:
            failed += 1
            print(f"[FAIL] {file_url} ({exc})", file=sys.stderr)

    print(
        f"[DONE] downloaded={downloaded} skipped={skipped} failed={failed} output={output_dir}"
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
